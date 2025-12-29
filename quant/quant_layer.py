import logging
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import List, Union
from quant.quantizer import AdaRoundQuantizer, ActivationQuantizer

QMODE = Enum('QMODE', ('QDIFF', 'NORMAL', 'PTQD'))
QMAP = {
        nn.Linear: F.linear,
        nn.Conv1d: F.conv1d,
        nn.Conv2d: F.conv2d,
    }

logger = logging.getLogger(__name__)

class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class QuantLayer(nn.Module):
    def __init__(self,
                 layer: Union[nn.Conv2d, nn.Linear, nn.Conv1d],
                 wq_params: dict = {},
                 aq_params: dict = {}, 
                 disable_aq: bool = False,
                 aq_mode: List[int] = [QMODE.QDIFF.value],
                 use_scale=False,
                 bound_range=1.0,
                 ) -> None:
        super().__init__()
        self.wq_params = wq_params
        self.aq_params = aq_params

        self.use_wq = False
        self.use_aq = False
        self.disable_aq = disable_aq
        self.ignore_recon = False
        self.aq_mode = aq_mode
        self.split = 0

        self.weight = layer.weight
        self.bias = layer.bias

        self.init = False
        self.use_scale = use_scale
        self.bound_range = bound_range

        self.fwd_kwargs = {}
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
            self.fwd_kwargs = dict(
                stride=layer.stride,
                padding=layer.padding, # 0,
                dilation=layer.dilation,
                groups=layer.groups
            )

        self.fwd_func = self.QMAP[type(layer)]

        self.wqtizer = AdaRoundQuantizer(w=self.weight.data.clone(), **self.wq_params)
        self.aqtizer = ActivationQuantizer(**self.aq_params)

        w_shape = [1, -1] + (len(self.weight.size()) - 2) * [1]
        self.gamma = nn.Parameter(torch.ones(self.weight.shape[1]).view(*w_shape))
        
        self.extra_repr = layer.extra_repr
    
    def forward(self, x, t=None, prev_emb_out=None, split=0) -> torch.Tensor:
        if not self.init:
            self.split = split
            if self.split != 0:
                logger.info(f'split: {split}')
                self.wqtizer = AdaRoundQuantizer(w=self.weight[:, :self.split], **self.wq_params)
                self.wqtizer1 = AdaRoundQuantizer(w=self.weight[:, self.split:], **self.wq_params)
                self.aqtizer1 = ActivationQuantizer(**self.aq_params)
            self.init = True
        
        x = x / self.gamma if self.use_scale else x
        w = self.weight * self.gamma if self.use_scale else self.weight
        b = self.bias

        if self.use_aq and not self.disable_aq:
            if self.split != 0:
                x = torch.cat([
                    self.aqtizer(x[:, :self.split], t), 
                    self.aqtizer1(x[:, self.split:], t)], 
                    dim=1
                )
            else:
                x = self.aqtizer(x, t)
        
        if self.use_wq:
            if self.split != 0:
                w = torch.cat([
                    self.wqtizer(w[:, :self.split]), 
                    self.wqtizer1(w[:, self.split:])], 
                    dim=1
                )
            else:
                w = self.wqtizer(w)

        if torch.isnan(x.sum()) or torch.isnan(w.sum()):
            logger.info("[QuantLayer] out is NAN, stopping training")
            pdb.set_trace()
    
        x = self.fwd_func(x, w, b, **self.fwd_kwargs)

        return x
    
    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        self.use_wq = use_wq if not self.ignore_recon else False
        self.use_aq = use_aq if not self.ignore_recon else False

    def set_running_stat(self,
                         running_stat: bool
                         ) -> None:
        self.aqtizer.running_stat = running_stat
