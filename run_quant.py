import argparse, os, sys, glob, datetime, time, yaml, math, json, gc
import pandas as pd
import numpy as np
import logging
import torch
import pdb

from torch import autocast
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from sample_ldm import generate_samples_ldm, generate_samples_ldm_imagenet, generate_samples_txt2img
from quant.calibration import cali_model, load_cali_model
from quant.data_generate import generate_cali_data_ldm, generate_cali_data_ldm_imagenet, generate_cali_text_guided_data
from quant.quant_layer import QMODE
from quant.quant_model import QuantModel
from quant.reconstruction_util import RLOSS
from quant.quant_utils import Scaler


def get_prompts(path: str,
                num: int = 128):
    '''
        COCO-Captions dataset
    '''
    df = pd.DataFrame(json.load(open(path))['annotations'])
    ps = df['caption'].sample(num).tolist()
    return ps


def prompts4eval(path: str,
                 batch_size: int = 1):
    df = pd.read_csv(path)
    prompts = df['caption'].tolist()
    res = []
    for i in range(math.ceil(len(prompts) / batch_size)):
        if (i + 1) * batch_size > len(prompts):
            res.append(prompts[i * batch_size:])
        else:
            res.append(prompts[i * batch_size:(i + 1) * batch_size])
    return res


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt):
    if ckpt:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
    else:
        pl_sd = {"state_dict": None}
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    del pl_sd
    gc.collect()
    torch.cuda.empty_cache()
    return model


def build_model(opt, ckpt):
    model = load_model(config, ckpt).to(memory_format=torch.channels_last)
    if getattr(model, 'model_ema', False):
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
    
    opt.q_mode = [QMODE.NORMAL.value]
    opt.asym = True
    opt.running_stat = True
    wq_params = {"bits": opt.w_bit,
                 "channel_wise": True,
                 "scaler": Scaler.MSE if opt.run_quant else Scaler.MINMAX,
                 "init": True}
                #  "init": True if not opt.load_quant else False}
    aq_params = {"bits": opt.a_bit,
                 "channel_wise": False,
                 "use_aq": opt.use_aq,
                 "dynamic": opt.dynamic,
                 "total_steps": opt.cali_steps,
                 "symmetric": opt.aq_symmetric,
                 "ratio_threshold": opt.ratio_threshold}
    if opt.ptq:
        # opt.run_quant &= not opt.load_quant # if load_quant, no need to run quant
        logger.info("Building and initializing Quantization model")
        with torch.no_grad():
            qnn = QuantModel(
                model=model.model.diffusion_model,
                wq_params=wq_params,
                aq_params=aq_params,
                softmax_a_bit=opt.softmax_a_bit,
                aq_mode=opt.q_mode,
                use_scale=opt.use_scale,
                use_split=opt.use_split
            )
        qnn.to('cuda').eval().to(memory_format=torch.channels_last)

        if opt.load_quant:
            logger.info("Loading quantized model...")
            load_aq = not opt.run_quant # only load weight quantization params. when running quantization

            shape = [opt.config.model.params.channels, opt.config.model.params.image_size, opt.config.model.params.image_size]
            if opt.task == 'txt2img':
                shape = [4, 64, 64]
            temp_data = (torch.randn(shape).unsqueeze(0), torch.randint(0, 1000, (1,)))
            if opt.task == 'ldm':
                pass
            elif opt.task == 'ldm_imagenet':
                uc_t = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(1 * [1000]).to(model.device)})
                temp_data += (uc_t,)
            elif opt.task == 'txt2img':
                temp_data += (torch.randn(1, 77, 768),)
            else:
                raise NotImplementedError(f"{opt.task} not supported yet!")
            load_cali_model(qnn, temp_data, use_aq=load_aq, path=opt.load_quant)

        if opt.run_quant:
            if os.path.exists(opt.cali_data_path):
                logger.info("Loading generated calibration data...")
                cali_data = np.load(opt.cali_data_path)
                cali_data = [torch.Tensor(cali_data[k]) for k in list(cali_data.keys())]
                cali_data[0] = cali_data[0].to(memory_format=torch.channels_last)
            else:
                logger.info("Generating calibration data...")
                if opt.task == 'ldm':
                    generate_cali_data = generate_cali_data_ldm
                    kwargs = dict(sampler=opt.sampler, num_samples=opt.cali_n_samples, batch_size=64)
                elif opt.task == 'ldm_imagenet':
                    generate_cali_data = generate_cali_data_ldm_imagenet
                    kwargs = dict(num_classes=opt.cali_n_classes, num_samples=opt.cali_n_samples, batch_size=32)
                elif opt.task == 'txt2img':
                    generate_cali_data = generate_cali_text_guided_data
                    kwargs = dict(
                        sampler=opt.sampler,
                        prompts=get_prompts(opt.caption_data_path),
                        precision_scope=autocast,
                        batch_size=1
                    )
                else:
                    raise NotImplementedError(f"{opt.task} not supported yet!")
                
                if opt.task == 'txt2img':
                    shape = [4, 64, 64]
                else:
                    shape = [
                        config.model.params.channels, 
                        config.model.params.image_size, 
                        config.model.params.image_size
                    ]
                cali_data = generate_cali_data(
                    model=load_model(config, ckpt).to(memory_format=torch.channels_last),
                    T=opt.cali_steps,
                    c=1,
                    shape=shape,
                    eta=opt.eta,
                    **kwargs
                )
                logger.info(cali_data[0].shape)
                os.makedirs(os.path.dirname(opt.cali_data_path), exist_ok=True)
                if len(cali_data) == 2:
                    np.savez(
                        opt.cali_data_path, 
                        x=cali_data[0].cpu().numpy(),
                        timesteps=cali_data[1].cpu().numpy(),
                    )
                else:
                    np.savez(
                        opt.cali_data_path, 
                        x=cali_data[0].cpu().numpy(),
                        timesteps=cali_data[1].cpu().numpy(),
                        context=cali_data[2].cpu().numpy(),
                    )
                logger.info(f"Calibration data saved at {opt.cali_data_path}")

            if getattr(model, 'first_stage_model', False):
                del model.first_stage_model
                model.first_stage_model = None
            if getattr(model, 'cond_stage_model', False):
                del model.cond_stage_model
                model.cond_stage_model = None

            gc.collect()
            torch.cuda.empty_cache()

            kwargs = dict(cali_data=cali_data,
                            iters=20000,
                            iters_scale=opt.iters_scale,
                            batch_size=opt.batch_size,
                            w=0.01,
                            asym=opt.asym,
                            warmup=0.2,
                            opt_mode=RLOSS.MSE,
                            multi_gpu=False,
                            load_quant=opt.load_quant,
                            amp_enabled=opt.amp_enabled,
                            layerwise_recon=opt.layerwise_recon,
                            ptf_layers=opt.ptf_layers,
                            r=opt.r,
                            momentum=opt.momentum,
                            task=opt.task,
            )
            qnn.to('cuda').eval().to(memory_format=torch.channels_last)
            qnn, cali_ckpt = cali_model(qnn=qnn,
                        use_aq=opt.use_aq,
                        path=opt.quant_save_path,
                        running_stat=opt.running_stat,
                        interval=opt.cali_n_samples * 2 if opt.task == 'ldm_imagenet' else opt.cali_n_samples,
                        lr=opt.lr,
                        lr_les=opt.lr_les,
                        **kwargs)
            
            logger.info(f"Maximum memory reserved: {torch.cuda.max_memory_reserved()  / 2**20} MB")
                
        model.model.diffusion_model = qnn
        model.model.ckpt = cali_ckpt
        model.model.iter = 0
            
        exit(0)
        
    return model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str,  default='ldm_imagenet', nargs="?", help="which task ? ['ldm', 'ldm_imagenet', 'txt2img']")
    parser.add_argument("--seed", type=int, default=123, help="the seed (for reproducible sampling)")
    parser.add_argument("-l", "--logdir", type=str, nargs="?", help="extra logdir", default="./results")
    parser.add_argument("-r", "--resume", type=str, nargs="?", help="load from logdir or checkpoint in logdir")

    # Sampling config
    parser.add_argument("--sampler", type=str, default='ddim', help="Which sampler to use (ddim, plms, dpm)",)
    parser.add_argument("-e", "--eta", type=float, nargs="?", help="eta for ddim sampling (0.0 yields deterministic sampling)", default=1.0)
    parser.add_argument("-n", "--n_samples", type=int, default=10, help="how many samples to produce for each given class. A.k.a. batch size",)
    parser.add_argument("-c", "--custom_steps", type=int, nargs="?", help="number of steps for ddim and fastdpm sampling", default=20)
    parser.add_argument("--scale", type=float, default=3.0, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)
    parser.add_argument("--classes", type=str, default="7,489,765", help="comma-separated list of classes to sample from")
    parser.add_argument("--batch_size", type=int, nargs="?", help="the bs", default=32)
    parser.add_argument("--amp_enabled", action="store_true", help="Train with amp", default=False)
    
    # quantization config
    parser.add_argument("--ptq", action="store_true", help="apply post-training quantization")
    parser.add_argument("--run_quant", action="store_true", help="perform quantization")
    parser.add_argument("--load_quant", type=str, default=None, help="load the quantized results")

    parser.add_argument('--use_aq',action='store_true',help='whether to use activation quantization')
    parser.add_argument("--w_bit", type=int, default=8)
    parser.add_argument("--a_bit", type=int, default=8)
    parser.add_argument("--softmax_a_bit", type=int, default=8, help="attn softmax activation bit")
    parser.add_argument("--lr", type=float, default=1e-3, help="attn softmax activation bit")
    parser.add_argument("--lr_les", type=float, default=1e-3, help="Learning rate for learned equivalent scaling")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic quantization for activation")
    parser.add_argument("--aq_symmetric", action="store_true", help="Symmetric quantization for activation")
    parser.add_argument("--use_scale", action="store_true", help="Scale weight and activation")
    parser.add_argument("--use_split", action="store_true", help="Split quantization params for skip-connection")
    parser.add_argument("--iters_scale", type=int, default=6000, help="Iterations for scale learning")
    parser.add_argument("--layerwise_recon", action="store_true", help="Layerwise reconstruction for Scale Learning")
    parser.add_argument("--loss_weight_type", type=str, default='focal', help="Timestep weighting loss type ['focal','linear']")
    parser.add_argument("--r", type=float, default=0, help="Power factor for LossFuncTimeWeighted")
    parser.add_argument("--ratio_threshold", type=float, default=0.85, help="Threshold of ratio of ptf frequent index")
    parser.add_argument("--ptf_layers", type=str, default='all', help="comma-separated list of layers for ptf ['in_layers','out_layers','op','skip_connection','qkv','proj_out','conv]")
    parser.add_argument("--momentum", type=float, default=0.95, help="momentum for update time-aware weight for LossFuncTimeWeighted")

    parser.add_argument("--cali_steps", type=int, default=20, help="Calibration data sampling step size")
    parser.add_argument("--cali_n_samples", type=int, default=256, help="Calibration data sample size")
    parser.add_argument("--cali_n_classes", type=int, default=32, help="Calibration data sampling step size")
    parser.add_argument("--cali_data_path", type=str, default=None, help="Calibration data path")
    parser.add_argument("--caption_data_path", type=str, default=None, help="Prompt data path for calibration data (just for txt2img)")
    
    # multi-gpu configs
    parser.add_argument('--multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3367', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # fix random seed
    seed_everything(opt.seed)
    opt.ngpus_per_node = torch.cuda.device_count()
    opt.world_size = opt.ngpus_per_node * opt.world_size
    # torch.backends.cudnn.deterministic = True

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    
    ckpt = opt.resume
    base_configs = sorted(glob.glob(os.path.join('/'.join(opt.resume.split('/')[:-1]), "config.yaml")))
    opt.base = base_configs

    # Setup configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # Setup result directories
    logdir = os.path.join(opt.logdir, opt.resume.split('/')[-2], now)
    if opt.cali_data_path == None:
        opt.cali_data_path = os.path.join(logdir, 'cali_data.npz')
    opt.quant_save_path = os.path.join(logdir, 'quant_sd.pth')

    os.makedirs(logdir, exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(logdir, 'run.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(75 * "=")
    logger.info("logging to:")
    logger.info(logdir)
    logger.info(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)

    opt.config = config
     # Run quantization
    model = build_model(opt, ckpt)
    model = model.cuda()

    # Generate samples
    if opt.task == 'ldm':
        generate_samples = generate_samples_ldm
        kwargs = dict(
            n_samples=opt.n_samples, 
            batch_size=opt.batch_size
        )
    elif opt.task == 'ldm_imagenet':
        generate_samples = generate_samples_ldm_imagenet
        kwargs = dict(
            batch_size=opt.batch_size,
            scale=opt.scale, 
            classes=opt.classes, 
            n_samples_per_class=min(opt.n_samples, 50)
        )
    elif opt.task == 'txt2img':
        generate_samples = generate_samples_txt2img
        kwargs = dict(
            scale=opt.scale,
            data=prompts4eval(opt.caption_data_path, opt.batch_size),
            sampler=opt.sampler,
        )
    else:
        raise NotImplementedError(f"{opt.task} not supported yet!")

    generate_samples(
        model, 
        logdir=logdir, 
        custom_steps=opt.custom_steps, 
        eta=opt.eta,
        **kwargs
    )

    logger.info("done.")
