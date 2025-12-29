import argparse, os, sys, glob, datetime, yaml, math, json
from typing import Tuple
import torch
import time
import random
import numpy as np
import pandas as pd
import logging
from tqdm import trange, tqdm

from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler

from quant.quant_layer import QMODE
from quant.quant_utils import Scaler
from quant.quantizer import RMODE
from quant.quant_model import QuantModel
from quant.calibration import load_cali_model

logger = logging.getLogger(__name__)


def prompts4eval(path: str,
                 batch_size: int = 1):
    df = pd.DataFrame(json.load(open(path))['annotations'])
    prompts = df['caption'].tolist()
    res = []
    for i in trange(math.ceil(len(prompts) / batch_size)):
        if (i + 1) * batch_size > len(prompts):
            res.append(prompts[i * batch_size:])
        else:
            res.append(prompts[i * batch_size:(i + 1) * batch_size])
    return res


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape,
                                        make_prog_row=True)
    else:
        sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                eta=eta)

    t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    return log


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def generate_samples_ldm(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    saved = False
    if model.cond_stage_model is None:
        all_images = []

        logger.info(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(math.ceil(n_samples / batch_size), desc="Sampling Batches (unconditional)"):
            if (_ + 1) * batch_size > n_samples:
                assert _ == math.ceil(n_samples / batch_size) - 1
                batch_size = n_samples - _ * batch_size
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            if not saved:
                os.makedirs(os.path.join(logdir, 'images'))
                save_logs(logs, os.path.join(logdir, 'images'), key="sample")
                saved = True
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                logger.info(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(logdir, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

    return all_img


def generate_samples_ldm_imagenet(model, logdir, batch_size, custom_steps, eta=0.0, scale=3.0, classes="", n_samples_per_class=3, n_samples=50000):
    sampler = DDIMSampler(model)

    if classes == "all":
        classes = list(range(1000))
        class_labels = random.choices(range(1000), k=n_samples)
    else:
        classes = [int(c) for c in classes.split(",")]
        class_labels = []
        for c in classes:
            class_labels += [c] * n_samples_per_class

    tstart = time.time()
    saved = False
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
                )
            base_count = 0
            all_imags = []
            all_labels = []

            # for class_label in classes:
            for i in trange(0, len(class_labels), batch_size):
                xc = torch.tensor(class_labels[i:i+batch_size])
                B = xc.size(0) # Due to last batch
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=custom_steps,
                                                conditioning=c,
                                                batch_size=B,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc[:B], 
                                                eta=eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                
                if not saved:
                    logger.info("Saving first batch")
                    os.makedirs(os.path.join(logdir, 'images'), exist_ok=True)
                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        imgpath = os.path.join(logdir, 'images', f"sample_{base_count:06}.png")
                        img.save(imgpath)
                        base_count += 1

                sample = 255. * rearrange(x_samples_ddim, 'b c h w -> b h w c').cpu().numpy()
                sample = sample.astype(np.uint8)
                all_imags.extend([sample])
                all_labels.extend([xc.cpu().numpy()])

    all_img = np.concatenate(all_imags, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    shape_str = "x".join([str(x) for x in all_img.shape])
    nppath = os.path.join(logdir, f"{shape_str}-samples.npz")
    np.savez(nppath, all_img, all_labels)

    logger.info(f"sampling of {n_samples_per_class * len(classes)} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

    return all_img

def generate_samples_txt2img(model, logdir, sampler, custom_steps, eta, scale, data, batch_size=1, n_samples=50000):
    if sampler == 'dpm':
        sampler = DPMSolverSampler(model)
    elif sampler == 'plms':
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    start_code = None
    saved = False
    with torch.no_grad():
        with torch.autocast("cuda"):
            with model.ema_scope():
                all_images= []
                all_prompts = []
                base_count = 0
                data = random.choices(data, k=n_samples//batch_size)
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    all_prompts += prompts
                    c = model.get_learned_conditioning(prompts)
                    samples_ddim, _ = sampler.sample(S=custom_steps,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=[4, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=eta,
                                                    x_T=start_code)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image = x_samples_ddim

                    if base_count < 64:
                        logger.info("Saving first batch")
                        os.makedirs(os.path.join(logdir, 'images'), exist_ok=True)
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * x_sample
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            imgpath = os.path.join(logdir, 'images', f"sample_{base_count:06}.png")
                            img.save(imgpath)
                            base_count += 1
                        saved = True
                    imags = 255. * x_checked_image
                    imags = imags.astype(np.uint8)
                    all_images.extend([imags])

                toc = time.time()

                all_img = np.concatenate(all_images, axis=0)
                shape_str = "x".join([str(x) for x in all_img.shape])
                nppath = os.path.join(logdir, f"{shape_str}-samples.npz")
                np.savez(nppath, all_img)

                with open(os.path.join(logdir, 'prompts.txt'), "w") as file:
                    for line in all_prompts:
                        file.write(line.strip() + "\n")
    
    return all_img


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

    return model


def build_model(opt, ckpt):
    model = load_model(opt.config, ckpt)
    if getattr(model, 'model_ema', False):
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
    
    opt.q_mode = [QMODE.NORMAL.value, QMODE.QDIFF.value]
    opt.asym = True
    opt.running_stat = True
    wq_params = {"bits": opt.w_bit,
                 "channel_wise": True,
                 "scaler": Scaler.MINMAX}
    aq_params = {"bits": opt.a_bit,
                 "channel_wise": False,
                 "scaler": Scaler.MINMAX,
                 "use_aq": opt.use_aq,
                 "symmetric": opt.aq_symmetric,
                 "ratio_threshold": opt.ratio_threshold}
    logger.info(aq_params)

    if opt.ptq:
        shape = [opt.config.model.params.channels, opt.config.model.params.image_size, opt.config.model.params.image_size]
        cali_data = (torch.randn(shape).unsqueeze(0), torch.randint(0, 1000, (1,)))
        if opt.task == 'ldm':
            pass
        elif opt.task == 'ldm_imagenet':
            uc_t = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(1 * [1000]).to(model.device)})
            cali_data += (uc_t,)
        elif opt.task == 'txt2img':
            cali_data += (torch.randn(1, 77, 768),)
        else:
            raise NotImplementedError(f"{opt.task} not supported yet!")

        logger.info("Loading quantized model [Ours] ...")
        wq_params['init'] = False
        wq_params["rmode"] = RMODE.NEAREST if opt.round_mode == 'nearest' else RMODE.LEARNED_HARD_SIGMOID
        aq_params['init'] = False
        aq_params['dynamic'] = opt.dynamic
        aq_params['total_steps'] = opt.custom_steps
        
        qnn = QuantModel(
            model=model.model.diffusion_model,
            wq_params=wq_params,
            aq_params=aq_params,
            cali=False,
            softmax_a_bit=opt.softmax_a_bit,
            aq_mode=opt.q_mode,
            use_scale=opt.use_scale,
            use_split=opt.use_split,
            bound_range=opt.bound_range
        )
        qnn.to('cuda').eval()
        
        setattr(model.model.diffusion_model, "split", True)
        load_cali_model(qnn, cali_data, use_aq=opt.use_aq and not opt.dynamic, path=opt.load_quant)
        qnn.set_quant_state(use_wq=True, use_aq=opt.use_aq)
        model.model.diffusion_model = qnn
        
        cali_ckpt = torch.load(opt.load_quant)
        tot = len(list(cali_ckpt.keys())) - 1
        if 'activation' in list(cali_ckpt.keys()):
            tot -= 1
        if tot > 1:
            model.model.tot = 1000 // tot
            model.model.t_max = tot - 1
            model.model.ckpt = cali_ckpt
            model.model.iter = 0

    return model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str,  default='ldm_imagenet', nargs="?", help="which task ? ['ldm', 'ldm_imagenet', 'txt2img']")
    parser.add_argument("-l", "--logdir", type=str, nargs="?", help="extra logdir", default="./results")
    parser.add_argument("-r", "--resume", type=str, nargs="?", help="load from logdir or checkpoint in logdir")
    parser.add_argument("--seed", type=int, default=123, help="the seed (for reproducible sampling)")

    # Sampling config
    parser.add_argument("--sampler", type=str, default='ddim', help="Which sampler to use (ddim, plms, dpm)",)
    parser.add_argument("-e", "--eta", type=float, nargs="?", help="eta for ddim sampling (0.0 yields deterministic sampling)", default=1.0)
    parser.add_argument("-n", "--n_samples", type=int, default=10000, help="how many samples to produce for each given class. A.k.a. batch size",)
    parser.add_argument("-c", "--custom_steps", type=int, default=20, nargs="?", help="number of steps for ddim and fastdpm sampling")
    parser.add_argument("--scale", type=float, default=3.0, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)
    parser.add_argument("--classes", type=str, default="all", help="comma-separated list of classes to sample from")
    parser.add_argument("--batch_size", type=int, nargs="?", help="the bs", default=64)
    
    # quantization config
    parser.add_argument("--ptq", action="store_true", help="apply post-training quantization")
    parser.add_argument("--load_quant", type=str, default=None, help="load the quantized results")

    parser.add_argument('--use_aq',action='store_true',help='whether to use activation quantization')
    parser.add_argument("--w_bit", type=int, default=8)
    parser.add_argument("--a_bit", type=int, default=8)
    parser.add_argument("--softmax_a_bit", type=int, default=8, help="attn softmax activation bit")
    parser.add_argument("--lr", type=float, default=1e-3, help="attn softmax activation bit")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic quantization for activation")
    parser.add_argument("--aq_symmetric", action="store_true", help="Symmetric quantization for activation")
    parser.add_argument("--use_scale", action="store_true", help="Symmetric quantization for activation")
    parser.add_argument("--use_split", action="store_true", help="Split quantization params for skip-connection")
    parser.add_argument("--bound_range", type=float, default=1.0, help="Bound range of activation")
    parser.add_argument("--round_mode", type=str, default='learned_hard_sigmoid', help="Round mode of weight quantizer")
    parser.add_argument("--ratio_threshold", type=float, default=0.85, help="ratio of threshold for PoT")

    parser.add_argument("--data_path", type=str, default=None, help="Prompt data path for calibration data (just for txt2img)")
    
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
    if opt.load_quant is not None:
        logdir += "_" + opt.load_quant.split("/")[-2].split("_")[-1]
    # logdir = os.path.join(opt.logdir, opt.resume.split('/')[-2], now + "_" + opt.load_quant.split("/")[-2].split("_")[-1])
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
    logger.info(sampling_conf)

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
            n_samples_per_class=min(opt.n_samples, 50),
            n_samples=opt.n_samples
        )
    elif opt.task == 'txt2img':
        generate_samples = generate_samples_txt2img
        kwargs = dict(
            scale=opt.scale,
            data=prompts4eval(opt.data_path, opt.batch_size),
            sampler=opt.sampler,
            batch_size=opt.batch_size,
            n_samples=opt.n_samples
        )
    else:
        raise NotImplementedError(f"{opt.task} not supported yet!")
    # pdb.set_trace()

    all_img = generate_samples(
        model, 
        logdir=logdir, 
        custom_steps=opt.custom_steps, 
        eta=opt.eta,
        **kwargs
    )

    logger.info("done.")
