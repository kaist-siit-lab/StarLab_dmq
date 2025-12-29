from typing import Any, Dict, Tuple, Union
import functools
from collections import defaultdict
import numpy as np
from tqdm import trange
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
import random
import gc

from ldm.modules.diffusionmodules.util import timestep_embedding

from models.quant_ldm_blocks import QuantQKMatMul, QuantSMVMatMul, QuantResBlock, QuantBasicTransformerBlock
from quant.quant_model import QuantModel
from quant.quant_layer import QuantLayer
from quant.quantizer import AdaRoundQuantizer, ActivationQuantizer, RMODE
from quant.utils import get_op_by_name, get_op_name, set_op_by_name
from quant.reconstruction import block_reconstruction, layer_reconstruction
import logging
import pdb

logger = logging.getLogger(__name__)


def cali_model(qnn: QuantModel,
                      cali_data: Tuple[torch.Tensor],
                      use_aq: bool = False,
                      path: str = None,
                      running_stat: bool = False,
                      interval: int = 128,
                      lr=1e-3,
                      lr_les=1e-3,
                      iters=20000,
                      iters_scale=6000,
                      asym=True,
                      load_quant=None,
                      layerwise_recon=False,
                      ptf_layers='all',
                      task='ldm',
                      **kwargs
                      ) -> None:
    logger.info("Calibrating...")

    amp_enabled = kwargs.get('amp_enabled', False)

    # Process ptf_layers argument
    if ptf_layers == 'all':
        ptf_layers = ['in_layers', 'out_layers', 'op', 'skip_connection', 'qkv', 'proj_out', 'conv']
    elif ptf_layers == '':
        ptf_layers = []
    else:
        ptf_layers = ptf_layers.split(',')

    print(f"PTF layers selected for calibration: {ptf_layers}")

    def recon_model(
            model, 
            lr=1e-3,
            iters=20000,
            use_wq=True, 
            use_aq=False,
            keep_gpu=False,
            asym=True,
            layer_wise=False,
    ):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        kwargs['interval'] = interval
        for name, module in model.named_children():
            full_name = get_op_name(qnn, module)
            logger.info(f"Processing module: {full_name}")
            keep_gpu = False

            # Skip specific non-quantized modules
            if isinstance(module, (QuantQKMatMul, QuantSMVMatMul)):
                continue
            
            # Apply layer-wise reconstruction
            elif isinstance(module, QuantLayer): 
                if not module.ignore_recon:
                    logger.info('Reconstruction for layer {}'.format(full_name))
                    layer_reconstruction(qnn, module, cali_data=cali_data, lr=lr_les, iters=iters, 
                        use_wq=use_wq, use_aq=use_aq, keep_gpu=False, asym=asym, **kwargs)
                
            elif isinstance(module, (QuantResBlock, QuantBasicTransformerBlock)) and not layer_wise:
                if not module.ignore_recon:
                    logger.info('Reconstruction for block {}'.format(full_name))
                    block_reconstruction(qnn, module, cali_data=cali_data, lr=lr_les, iters=iters, 
                        use_wq=use_wq, use_aq=use_aq, keep_gpu=keep_gpu, asym=asym, **kwargs)
                
            else:
                recon_model(module, lr=lr, iters=iters, use_wq=use_wq, use_aq=use_aq, 
                            keep_gpu=keep_gpu, asym=asym, layer_wise=layer_wise)
        return True

    if load_quant:
        logger.info("Skip weight calibration by loading weights quantization parameters.")
        wq_dict = torch.load(load_quant, map_location='cpu')['weight']
        model_dict = {'weight': wq_dict}

        # Initialize the model and perform a forward pass to set up quantizers
        with torch.no_grad():
            qnn.cuda().eval()
            inputs = [x[:8].cuda() for x in cali_data]
            _ = qnn(*inputs)
    else:
        wq_dict = dict()
        ################################ Scale calibration ################################
        if qnn.use_scale:
            logger.info("Starting scale calibration...")
            
            # Initialize the model and perform a forward pass to set up quantizers
            with torch.no_grad():
                qnn.cuda().eval()
                inputs = [x[:8].cuda() for x in cali_data]
                _ = qnn(*inputs)

            # Set weight quantizers to dynamic mode for scale parameter learning
            for name, module in qnn.model.named_modules():
                if isinstance(module, AdaRoundQuantizer):
                    module.dynamic = True
                    module.rmode = RMODE.NEAREST_STE
            
            qnn.set_quant_state(use_wq = True, use_aq = True)
            recon_model(qnn, lr=lr, iters=iters_scale, use_wq=True, use_aq=True, 
                        asym=False, layer_wise=layerwise_recon)
            qnn.zero_grad(set_to_none=True)
            
            # Save learned scale parameters (gamma)
            for name, module in qnn.model.named_modules():
                if isinstance(module, QuantLayer) and module.gamma is not None:
                    wq_dict['model.' + name + '.gamma'] = module.gamma.cpu().data
            model_dict = {'weight': wq_dict}
            torch.save(model_dict, path.split('.')[0] + '_scale.pth')

            # Reset AdaRoundQuantizers to their learned rounding mode
            for name, module in qnn.model.named_modules():
                if isinstance(module, AdaRoundQuantizer):
                    module.dynamic = False
                    module.rmode = RMODE.LEARNED_HARD_SIGMOID
                # Detach delta and zero_point for ActivationQuantizers if present
                elif 'aqtizer' in name and module.delta is not None:
                    try:
                        module.delta = module.delta.detach()
                        module.zero_point = module.zero_point.detach()
                    except:
                        continue

            with torch.no_grad():
                qnn.cuda().eval()
                for name, module in qnn.model.named_modules():
                    if isinstance(module, QuantLayer) and module.gamma is not None:
                        module_weight = module.weight * module.gamma if module.use_scale else module.weight

                        if module.split == 0:
                            module.wqtizer.delta, module.wqtizer.zero_point = \
                                module.wqtizer._init_quantization_param(module_weight, 
                                                                        module.wqtizer.channel_wise)
                            module.wqtizer.init_alpha(x=module_weight)

                        else:
                            module.wqtizer.delta, module.wqtizer.zero_point = \
                                module.wqtizer._init_quantization_param(module_weight[:,:module.split], 
                                                                        module.wqtizer.channel_wise)
                            module.wqtizer.init_alpha(x=module_weight[:,:module.split])

                            module.wqtizer1.delta, module.wqtizer1.zero_point = \
                                module.wqtizer1._init_quantization_param(module_weight[:,module.split:], 
                                                                         module.wqtizer1.channel_wise)
                            module.wqtizer1.init_alpha(x=module_weight[:,module.split:])
            
            logger.info(f"Maximum memory reserved after scale init: {torch.cuda.max_memory_reserved() / 2**20:.2f} MB")
            torch.cuda.empty_cache()
            
        ############################### Weight calibration ###############################
            logger.info("Starting weight calibration...")
            # Re-initialize wqtizer params after potential scale adjustments
            with torch.no_grad():
                qnn.cuda().eval()
                inputs = [x[:8].cuda() for x in cali_data]
                _ = qnn(*inputs)
            
            qnn.set_quant_state(use_wq = True, use_aq = False)
            recon_model(qnn, lr=lr, iters=iters, use_wq=True, use_aq=False, asym=True)
            logger.info(f"Maximum memory reserved after weight calibration: {torch.cuda.max_memory_reserved() / 2**20:.2f} MB")

            # Save weight quantization parameters
            wq_dict.update(get_wqtizer_params_dict(qnn)) # Update with newly calibrated params
            model_dict = {'weight': wq_dict}
            torch.save(model_dict, path)

        ############################# Activation calibration #############################
    if use_aq:
        logger.info("Starting activation calibration...")

        for time in trange(cali_data[0].shape[0] // interval):
            t_cali_data = tuple([x[time * interval: (time + 1) * interval] for x in cali_data])

            # Reset activation quantizers before collecting min-max range
            qnn.set_quant_state(use_wq = True, use_aq = True)
            for name, module in qnn.model.named_modules():
                if 'aqtizer' in name:
                    # Delete existing delta/zero_point to force re-initialization
                    del module.delta
                    del module.zero_point
                    module.use_ptf = False
                    module.delta = None
                    module.zero_point = None

            # Get min-max range of activation (momentum)
            qnn.cuda().eval()
            batch_size = min(16, t_cali_data[0].shape[0])
            with torch.no_grad():
                # Randomly sample a batch for initial statistics
                inds = np.random.choice(t_cali_data[0].shape[0], 16, replace=False)
                inputs = (x[inds].cuda() for x in t_cali_data)
                _ = qnn(*inputs)

                if running_stat:
                    logger.info('Collecting running statistics for activation calibration...')
                    all_inds = np.arange(t_cali_data[0].shape[0])
                    np.random.shuffle(all_inds)
                    qnn.set_running_stat(True) # Enable running statistics for activation quantizers
                    for i in range(0, t_cali_data[0].shape[0], batch_size):
                        inputs = (x[all_inds[i: i + batch_size]].cuda() for x in t_cali_data)
                        _ = qnn(*inputs)
                    qnn.set_running_stat(False)
                    logger.info('Running statistics collection for activation calibration done.')

                # Set quant state for actual calibration after stats collection
                qnn.set_quant_state(use_wq=False, use_aq=True) # Focus on AQ calibration
                     
            if len(ptf_layers) != 0:
                # Determine batch size for PTF based on task due to model size
                if task == 'ldm_imagenet':
                    batch_size = 32
                elif task == 'txt2img':
                    batch_size = 8
                else:
                    batch_size = 64
                
                # Move model to CPU to free GPU memory for PTF data collection
                qnn.cpu().eval()
                gc.collect()
                torch.cuda.empty_cache()

                qnn.set_quant_state(use_wq = True, use_aq = True) # Enable both for PTF pass
                qnn.set_running_stat(True) # Re-enable running stat for PTF collection if needed

                with torch.no_grad() and autocast(enabled=amp_enabled):
                    inputs = [x.cuda() for x in t_cali_data]
                    x = inputs[0]
                    timesteps = inputs[1]
                    context = inputs[2] if len(inputs) == 3 else None

                    t_emb = timestep_embedding(timesteps, qnn.model.model_channels, repeat_only=False)
                    qnn.model.time_embed = qnn.model.time_embed.cuda()
                    emb = qnn.model.time_embed(t_emb)
                    qnn.model.time_embed = qnn.model.time_embed.cpu()

                    gc.collect()
                    torch.cuda.empty_cache()

                    hs = []
                    emb_outs = []
                    h = x.type(qnn.model.dtype)

                    # Input blocks
                    emb_out = None
                    for module in qnn.model.input_blocks:
                        h, emb_out = apply_ptf(module, h, emb, context, timesteps, emb_out, 
                                               ptf_layers=ptf_layers, bs=batch_size)
                        hs.append(h.cpu())
                        if emb_out is not None:
                            emb_outs.append(emb_out.cpu())
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    # Middle block
                    h, emb_out = apply_ptf(qnn.model.middle_block, h, emb, context, timesteps, emb_out, 
                                           ptf_layers=ptf_layers, bs=batch_size)

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Output blocks
                    for module in qnn.model.output_blocks:
                        # Handle skip connections and concatenated embeddings
                        split = h.shape[1] if qnn.use_split else 0
                        h = torch.cat([h, hs.pop().cuda()], dim=1)

                        if emb_outs:
                            emb_out = torch.cat([emb_out, emb_outs.pop().cuda()], dim=1)
                        
                        h, emb_out = apply_ptf(module, h, emb, context, timesteps, emb_out, 
                                               split=split, ptf_layers=ptf_layers, bs=batch_size)

                        gc.collect()
                        torch.cuda.empty_cache()

                    h = h.type(x.dtype)

                    if qnn.model.predict_codebook_ids:
                        qnn.model.id_predictor = qnn.model.id_predictor.cuda()
                        _ = qnn.model.id_predictor(h)
                        qnn.model.id_predictor = qnn.model.id_predictor.cpu()
                    else:
                        qnn.model.out = qnn.model.out.cuda()
                        _ = qnn.model.out(h)
                        qnn.model.out = qnn.model.out.cpu()
                    
                    gc.collect()
                    torch.cuda.empty_cache()

                qnn.set_running_stat(False)

            # Save activation quantization parameters for the current interval
            aq_dict = get_aqtizer_params_dict(qnn)
            model_dict['act_{}'.format(time)] = aq_dict

    if path:
        torch.save(model_dict, path)
    logger.info("Calibration done.")
    return qnn, model_dict


def get_ptf_qtizers(module, ptf_layers):
    ptf_qtizers = dict()
    for name, m in module.named_modules():
        for layer_name in ptf_layers:
            # Check if layer_name is in the module's full name and it's an aqtizer,
            # excluding "aqtizer_" to avoid internal quantizers.
            if layer_name in name and 'aqtizer' in name and 'aqtizer_' not in name:
                ptf_qtizers[name] = m
    return ptf_qtizers

def cache_input_hook(m, x, y, name, feat_dict, device='cpu'):
    x = x[0]
    x = x.detach().to(device)
    feat_dict[name].append(x)


@torch.no_grad()
def apply_ptf(module, x, emb, context, timesteps, emb_out, split=0, ptf_layers=[], bs=16):
    cache_device = 'cuda' if bs >= 128 else 'cpu'

    # Filter out None inputs
    inps = [arg for arg in [x, emb, context, timesteps, emb_out] if arg is not None]

    ptf_qtizers = get_ptf_qtizers(module, ptf_layers)
    
    input_feat = defaultdict(list)
    handles = []

    # Register forward hooks to cache activation inputs
    for name, quantizer_module in ptf_qtizers.items():
        handles.append(
            quantizer_module.register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat, device=cache_device)
            )
        )
    
    N = x.shape[0]
    module = module.cuda()

    # Collect inputs for activation quantizers by running a forward pass
    for idx in range(0, N, bs):
        inp_batch = [x[idx:idx + bs].cuda() for x in inps]
        _ = module(*inp_batch, split=split)
    
    # Remove hooks after collecting inputs
    for h in handles:
        h.remove()

    gc.collect()
    torch.cuda.empty_cache()

    # Concatenate collected inputs
    input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

    # Calculate power of two factors
    for name, qtizer in ptf_qtizers.items():
        x = input_feat[name].cuda()
        qtizer.use_ptf = True
        _, _ = qtizer.get_ptf_scale_zero_point(x, 4)
    
    # Get outputs of the current module with updated quantizers
    hs = []
    emb_outs = []
    for idx in range(0, N, bs):
        inp_batch = [x[idx:idx + bs].cuda() for x in inps]
        h, emb_out = module(*inp_batch, split=split)
        hs.append(h)
        emb_outs.append(emb_out)

    hs = torch.cat(hs, dim=0)
    emb_outs = torch.cat(emb_outs, dim=0) if emb_outs[0] is not None else None

    module = module.cpu()
    del input_feat
    gc.collect()
    torch.cuda.empty_cache()
    return hs, emb_outs


@torch.no_grad()
def get_wqtizer_params_dict(qnn):
    wq_dict = dict()
    for name, module in qnn.model.named_modules():
        if 'wqtizer' in name:
            wq_dict['model.' + name + '.delta'] = module.delta.cpu().data
            wq_dict['model.' + name + '.zero_point'] = module.zero_point.cpu().data
            wq_dict['model.' + name + '.alpha'] = module.alpha.cpu().data

            if getattr(module, 'scale_mask', None) is not None:
                wq_dict['model.' + name + '.scale_mask'] = module.scale_mask.cpu().data

    return wq_dict    


@torch.no_grad()
def get_aqtizer_params_dict(qnn):
    aq_dict = dict()

    param_names = [
        'delta', 'zero_point', 'delta_list', 
        'zero_point_list', 'scale_mask', 'index_ratio'
    ]

    for name, module in qnn.model.named_modules():   
        if isinstance(module, ActivationQuantizer):
            for param_name in param_names:
                param_value = getattr(module, param_name, None)
                if param_value is not None:
                    if isinstance(param_value, torch.Tensor):
                        aq_dict[f'model.{name}.{param_name}'] = param_value.cpu().data
                    else:
                        aq_dict[f'model.{name}.{param_name}'] = param_value
    return aq_dict


@torch.no_grad()
def load_cali_model(qnn: QuantModel,
                    cali_data=None,
                    use_aq: bool = False,
                    path: str = None,
                    ) -> None:
    logger.info(f"Loading calibration model from {path}...")
    ckpt = torch.load(path, map_location='cpu')
    w_ckpt = ckpt['weight']

    qnn.disable_out_quantization() # disable out first and last layer
    qnn.set_quant_state(use_wq = True, use_aq = False)
    _ = qnn(*(_.cuda() for _ in cali_data))
    
    for k, v in w_ckpt.items():
        if 'time_embed' in k or 'emb_layers' in k:
            continue
          
        if k.split('.')[-1] in ['beta', 'gamma']:
            v = nn.Parameter(v)
        elif k.split('.')[-1] in ['alpha']:
            v = nn.Parameter(v)
        try:
            set_op_by_name(qnn, k, v)
        except:
            print("Fail to load", k)
            continue

    if use_aq:
        if 'activation' in ckpt.keys():
            a_ckpt = ckpt['activation']

            for k, v in a_ckpt.items():
                v = nn.Parameter(v)
                set_op_by_name(qnn, k, v)
        
    logger.info("Loading calibration model done.")
