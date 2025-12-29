import torch 
from torch import nn
from quant.quant_utils import ste_round, mse, Scaler, lp_loss, REDUCTION
from enum import Enum
import pdb
import logging
logger = logging.getLogger(__name__)


RMODE = Enum('RMODE', ('LEARNED_ROUND_SIGMOID', 
                       'NEAREST', 
                       'NEAREST_STE', 
                       'STOCHASTIC',
                       'LEARNED_HARD_SIGMOID'))


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param w: initialize alpha
    :param bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param round_mode: controls the forward pass in this quantizer
    """
    def __init__(self,
                 w: torch.Tensor,
                 bits: int = 8,
                 symmetric: bool = False,
                 channel_wise: bool = False,
                 scaler: Scaler = Scaler.MINMAX,
                 rmode: RMODE = RMODE.LEARNED_HARD_SIGMOID,
                 init = True,
                 dynamic=False,
                 **kwargs,
                ) -> None:
        
        super().__init__()
        self.level = 2 ** bits
        self.symmetric = symmetric
        self.scaler = scaler
        self.channel_wise = channel_wise
        self.rmode = rmode
        self.soft_tgt = False 
        
        self.gamma, self.zeta = -0.1, 1.1
        self.alpha = None

        D = {4: (-1, 1, 1, 1), 3: (-1, 1, 1), 2: (-1, 1)}
        self.delta = torch.ones(w.shape[0]).view(D[len(w.shape)])
        self.zero_point = torch.zeros(w.shape[0]).view(D[len(w.shape)])
        self.alpha = torch.zeros_like(w)

        if init:
            self.delta, self.zero_point = self._init_quantization_param(w, self.channel_wise)
        self.init_alpha(x=w.clone())

        self.dynamic = dynamic

        if self.symmetric:
            self.min_int = -(2 ** (bits - 1))
            self.max_int = 2 ** (bits - 1) - 1
        else:
            self.min_int = 0
            self.max_int = 2 ** bits - 1

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.to(x.device)
        if isinstance(self.zero_point, torch.Tensor):
            self.zero_point = self.zero_point.to(x.device)
        
        if self.dynamic:
            delta, zero_point = self.get_scale_zero_point(x)
        else:
            delta, zero_point = self.delta, self.zero_point
        
        x_q = torch.clamp(self.round_func(x, delta) + zero_point, self.min_int, self.max_int)
        x_dq = delta * (x_q - zero_point)
        return x_dq
        
    def get_soft_tgt(self, gamma=-0.1, zeta=1.1) -> torch.Tensor:
        return torch.clamp(torch.sigmoid(self.alpha) * (zeta - gamma) + gamma, 0, 1)
    
    def round_func(self, x, delta):
        x_floor = torch.floor(x / delta)
        if self.rmode == RMODE.NEAREST:
            x_int  = torch.round(x / delta)
        elif self.rmode == RMODE.NEAREST_STE:
            x_int = ste_round(x / delta)
        elif self.rmode == RMODE.STOCHASTIC:
            x_int = x_floor + torch.bernoulli((x / delta) - x_floor)
        elif self.rmode == RMODE.LEARNED_HARD_SIGMOID:
            if self.soft_tgt:
                x_int = x_floor + self.get_soft_tgt().to(x.device)
            else:
                alpha = self.alpha.to(x.device)
                x_int = x_floor + (alpha >= 0).float()
        else:
            raise NotImplementedError
        return x_int
    
    def init_alpha(self, x: torch.Tensor) -> None:
        self.delta = self.delta.to(x.device)
        rest = (x / self.delta) - torch.floor(x / self.delta)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
        self.alpha = nn.Parameter(alpha)
    
    def _init_quantization_param(self, 
                                 x: torch.Tensor, 
                                 channel_wise: bool = False
                                 ) -> [torch.Tensor, torch.Tensor]:
        if channel_wise:
            N = x.shape[0]
            x_clone = x.clone().detach()
            x_max = x_clone.abs()
            for _ in range(len(x.shape) - 1):
                x_max = x_max.max(dim=-1)[0]
            delta, zero_point = x_max.clone(), x_max.clone()
            for c in range(N):
                delta[c], zero_point[c] = self._init_quantization_param(x_clone[c], channel_wise=False)
            D = {4: (-1, 1, 1, 1), 3: (-1, 1, 1), 2: (-1, 1)}
            delta = delta.view(D[len(x.shape)]) 
            zero_point = zero_point.view(D[len(x.shape)])
        else:
            delta, zero_point = self.scaler(x, self.symmetric, self.level, always_zero=False)
        return delta, zero_point
    
    def get_scale_zero_point(self, x):
        dim = [*range(1,len(x.shape))] if self.channel_wise else [*range(len(x.shape))]

        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)

        if self.symmetric:
            rng = torch.max(x_min.abs(), x_max.abs())
            delta = rng / self.max_int
            zero_point = torch.zeros(1, device=x.device)
        else:
            rng = x_max - x_min
            delta = rng / self.max_int
            # zero_point = -torch.round(x_min / delta).clamp_(self.min_int, self.max_int)
            zero_point = torch.round(- x_min / delta).clamp_(self.min_int, self.max_int)
        
        return delta, zero_point

    def extra_repr(self) -> str:
        s = 'level={}, symmetric={}, rmode={}'.format(self.level, self.symmetric, self.rmode)
        return s.format(**self.__dict__)


class ActivationQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, 
                 bits: int = 8,
                 symmetric: bool = False,
                 channel_wise: bool = False,
                 dynamic=True,
                 init=True,
                 ratio_threshold=1.0,
                 after_silu=False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.dynamic = dynamic
        self.running_stat = False
        self.init = init
        self.ratio_threshold = ratio_threshold
        self.after_silu = after_silu
        self.silu_min = -0.27846455574035645

        self.delta = None
        self.zero_point = None
        self.scale_mask = None

        if self.symmetric:
            self.min_int = -(2 ** (bits - 1))
            self.max_int = 2 ** (bits - 1) - 1
        else:
            self.min_int = 0
            self.max_int = 2 ** bits - 1
        
        self.x_min = None
        self.x_max = None
    
    def forward(self, x, t=None):
        x = x - self.silu_min if self.after_silu else x

        if self.init or self.dynamic or self.running_stat:
            self.delta, self.zero_point = self.get_scale_zero_point(x, running_stat=self.running_stat)
            self.init = False
        
        if self.scale_mask != None:
            x_shape = [1, -1] + (x.ndim - 2) * [1]
            delta = (self.delta * self.scale_mask.reshape(*x_shape))
        else:
            delta = self.delta

        x_q = torch.clamp(ste_round(x / delta) + self.zero_point, self.min_int, self.max_int)
        x_dq = delta * (x_q - self.zero_point)

        x_dq = x_dq + self.silu_min if self.after_silu else x_dq
        return x_dq

    def get_scale_zero_point(self, x, running_stat=False, momentum=0.95):
        dim = [0, *range(2,len(x.shape))] if self.channel_wise else [*range(len(x.shape))]

        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)

        if self.x_min is None:
            self.x_min, self.x_max = x_min.detach().clone(), x_max.detach().clone()
        
        # Get min/max with momentum
        if running_stat:
            x_min = self.x_min * momentum + x_min * (1. - momentum)
            x_max = self.x_max * momentum + x_max * (1. - momentum)
            self.x_min, self.x_max = x_min, x_max

        # Compute delta & zero_point
        if self.symmetric:
            rng        = torch.max(x_min.abs(), x_max.abs())
            delta      = rng / self.max_int
            zero_point = torch.zeros(1, device=x.device)
        else:
            rng        = x_max - x_min
            delta      = rng / self.max_int
            zero_point = torch.round(-x_min / delta)#.clamp_(self.min_int, self.max_int)
        
        return delta, zero_point

    def get_ptf_scale_zero_point(sefl, x, num_scales=4):
        D = x.ndim
        N = x.size(0)
        channel_dim = 1
        C = x.size(channel_dim)

        if self.delta is None:
            scale, zero_point = self.get_scale_zero_point(x, channel_wise=False, running_stat=False)
        else:
            scale, zero_point = self.delta, self.zero_point
        scale = scale.to(x.device)
        zero_point = zero_point.to(x.device)
        
        scales = [scale / (2 ** i) for i in range(num_scales)] # [scale8, scale4, scale2, scale1]
        scales.reverse()
  
    def extra_repr(self) -> str:
        s = 'bits={bits}, symmetric={symmetric}, channel_wise={channel_wise}, dynamic={dynamic}'
        return s.format(**self.__dict__)
