import copy
import gc
import logging
import math
import pdb
from typing import Tuple

import torch
from torch.cuda.amp import GradScaler, autocast

import linklink as link

from models.quant_ldm_blocks import BaseQuantBlock
from quant.quant_layer import QMODE, QuantLayer
from quant.quant_model import QuantModel
from quant.reconstruction_util import RLOSS, LossFunc, LossFuncTimeWeighted
from quant.data_utill import save_inout
from quant.utils import get_op_name, set_op_by_name

logger = logging.getLogger(__name__)


def layer_reconstruction(model: QuantModel,
                         layer: QuantLayer,
                         cali_data: Tuple[torch.Tensor],
                         batch_size: int = 128,
                         iters: int = 20000,
                         w: float = 0.001,
                         opt_mode: RLOSS = RLOSS.MSE,
                         asym: bool = False,
                         b_range: tuple = (20, 2),
                         warmup: float = 0.0,
                         use_wq: bool = True,
                         use_aq: bool = False,
                         lr: float = 1e-3,
                         p: float = 2.0,
                         r: float = 0.0,
                         loss_weight_type: str = 'focal',
                         momentum: float = 0.95,
                         multi_gpu: bool = False,
                         keep_gpu=True,
                         interval=256,
                         amp_enabled=False,
                         ) -> None:
    name = get_op_name(model, layer)
    
    model.set_quant_state(use_wq=False, use_aq=False)
    layer.set_quant_state(use_wq=use_wq, use_aq=use_aq)

    model.zero_grad(set_to_none=True)
    model = model.cuda()

    cached_inputs, cached_outputs = save_inout(model, layer, cali_data, asym, use_wq, use_aq, batch_size, keep_gpu)

    with torch.no_grad():
        layer_org = layer
        layer.zero_grad(set_to_none=True)
        layer = copy.deepcopy(layer)
        layer_org.set_quant_state(use_wq=False, use_aq=False)

    model = model.cpu()
    layer_org = layer_org.cuda()
    layer = layer.cuda()

    gc.collect()
    torch.cuda.empty_cache()

    device = next(layer.parameters()).device

    # Weight calibration
    if use_wq and not use_aq:
        layer.wqtizer.soft_tgt = True
        opt_params = [layer.wqtizer.alpha]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = None
        loss_func = LossFunc(
            o=layer,
            round_loss=RLOSS.NONE if use_aq else RLOSS.RELAXATION,
            w=w,
            max_count=iters,
            rec_loss=opt_mode,
            b_range=b_range,
            decay_start=0.0,
            warmup=warmup,
            p=p,
            r=r,
            momentum=momentum,
            loss_weight_type=loss_weight_type
        )
        
    # Time step activation (beta, gamma) calibration
    elif use_wq and use_aq:
        opt_params = [
            {"params": [layer.gamma], "lr": lr},
        ]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)
        # scheduler = None
        loss_func = LossFuncTimeWeighted(r=r, rec_loss=opt_mode, p=p, momentum=momentum, loss_weight_type=loss_weight_type)
    else:
        raise NotImplementedError(f"use_wq: {use_wq} use_aq: {use_aq} not supported yet!")
    
    scaler = GradScaler(enabled=amp_enabled)
    
    assert opt_mode == RLOSS.MSE

    N = cached_inputs[0].size(0)
    layer.train()
    for i in range(iters):
        idx = torch.randperm(N)[: batch_size]
        cur_inputs = [x[idx].to(device=device, non_blocking=True) for x in cached_inputs]

        cur_outputs = cached_outputs[idx].to(device=device, non_blocking=True)

        out_quant = layer(*cur_inputs) # ^z
        err = loss_func(out_quant, cur_outputs, cur_inputs[1])
        scaler.scale(err).backward() # (retain_graph=True)

        if multi_gpu:
            for param in opt_params: # output layer does not use quantizer
                if param.grad is not None:
                    link.allreduce(param.grad)

        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        optimizer.zero_grad()

    layer.zero_grad(set_to_none=True)
    layer.eval()
    layer.wqtizer.soft_tgt = False

    del opt_params, optimizer, scheduler, scaler
    del cached_inputs, cached_outputs
    del layer_org
    
    set_op_by_name(model, name, layer)

    gc.collect()
    torch.cuda.empty_cache()


def block_reconstruction(model: QuantModel,
                         block: BaseQuantBlock,
                         cali_data: torch.Tensor,
                         batch_size: int = 32,
                         iters: int = 20000,
                         w: float = 0.01,
                         opt_mode: RLOSS = RLOSS.MSE,
                         asym: bool = False,
                         b_range: tuple = (20, 2),
                         warmup: float = 0.0,
                         use_wq: bool = True,
                         use_aq: bool = False,
                         lr: float = 4e-5,
                         p: float = 2.0,
                         r: float = 0.0,
                         loss_weight_type: str = 'focal',
                         momentum: float = 0.95,
                         multi_gpu: bool = True,
                         keep_gpu=True,
                         interval=256,
                         amp_enabled=False,
                         ) -> None:
    # pdb.set_trace()
    full_name = get_op_name(model, block)

    model.set_quant_state(use_wq=False, use_aq=False)
    block.set_quant_state(use_wq=use_wq, use_aq=use_aq)

    model.zero_grad(set_to_none=True)
    model = model.cuda()

    cached_inputs, cached_outputs = save_inout(model, block, cali_data, asym, use_wq, use_aq, batch_size, keep_gpu)

    with torch.no_grad():
        block_org = block
        block = copy.deepcopy(block)
        block_org.set_quant_state(use_wq=False, use_aq=False)
    
    model = model.cpu()
    block_org = block_org.cuda()
    block = block.cuda()

    gc.collect()
    torch.cuda.empty_cache()

    # Weight calibration
    if use_wq and not use_aq:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer):
                module.wqtizer.soft_tgt = True
                opt_params += [module.wqtizer.alpha]

                if module.split != 0:
                    module.wqtizer1.soft_tgt = True
                    opt_params += [module.wqtizer1.alpha]

        if len(opt_params) == 0: # for QuantSMVMatMul and QuantQKMatMul
            return
        
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = None
        loss_func = LossFunc(
            o=block,
            round_loss=RLOSS.NONE if use_aq else RLOSS.RELAXATION,
            w=w,
            max_count=iters,
            rec_loss=opt_mode,
            b_range=b_range,
            decay_start=0.0,
            warmup=warmup,
            p=p,
            r=r,
            momentum=momentum,
            loss_weight_type=loss_weight_type
        )

    # Time step activation (beta, gamma) calibration
    elif use_wq and use_aq:
        opt_params0 = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer) and not module.disable_aq:
                opt_params0 += [module.gamma]
        opt_params = [
            {"params": opt_params0, "lr": lr}, 
        ]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)
        loss_func = LossFuncTimeWeighted(r=r, rec_loss=opt_mode, p=p, momentum=momentum, loss_weight_type=loss_weight_type)
    else:
        raise NotImplementedError(f"use_wq: {use_wq} use_aq: {use_aq} not supported yet!")
    
    scaler = GradScaler(enabled=amp_enabled)

    assert opt_mode == RLOSS.MSE

    device = next(block.parameters()).device
    N = cached_inputs[0].size(0)
    block.train()
    for i in range(iters):
        idx = torch.randperm(N)[: batch_size]
        cur_inputs = [x[idx].to(device=device, non_blocking=True) for x in cached_inputs]
        cur_outputs = cached_outputs[idx].to(device=device, non_blocking=True)

        # ResBlock's split or ResnetBlock's split has been set in save_inout or even before, and cur_inputs does not contain split
        out_quant = block(*cur_inputs)
        if isinstance(out_quant, (Tuple, list)):
            out_quant = out_quant[0]

        err = loss_func(out_quant, cur_outputs, cur_inputs[2])
        if not math.isfinite(err.item()):
            logger.info("Loss is NAN, stopping training")
            pdb.set_trace()

        scaler.scale(err).backward()

        if multi_gpu:
            for param in opt_params:
                link.allreduce(param.grad)

        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        optimizer.zero_grad()

    block.zero_grad(set_to_none=True)
    block.eval()

    for _, module in block.named_modules():
        if isinstance(module, QuantLayer):
            if getattr(module, 'split', None) == 0 and QMODE.QDIFF.value in module.aq_mode:
                module.wqtizer.soft_tgt = False
                module.wqtizer1.soft_tgt = False
            else:
                module.wqtizer.soft_tgt = False
    
    del opt_params, optimizer, scheduler, scaler
    del cached_inputs, cached_outputs
    del block_org

    set_op_by_name(model, full_name, block)

    gc.collect()
    torch.cuda.empty_cache()
