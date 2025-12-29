from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.diffusionmodules.openaimodel import (TimestepBlock, TimestepEmbedSequential, 
                                                      Upsample, Downsample, ResBlock, 
                                                      AttentionBlock, QKMatMul,SMVMatMul,)
from ldm.modules.attention import exists, default, CrossAttention, BasicTransformerBlock, SpatialTransformer
from quant.quant_layer import QuantLayer
from quant.quantizer import ActivationQuantizer


class QuantTimestepEmbedSequential(TimestepEmbedSequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def __init__(self, *args):
        super(QuantTimestepEmbedSequential, self).__init__(*args)

    def forward(self, x, emb, context=None, t=None, prev_emb_out=None, split=0):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x, prev_emb_out = layer(x, emb, t, prev_emb_out, split=split)
            elif isinstance(layer, QuantSpatialTransformer):
                x = layer(x, context, t=t, prev_emb_out=prev_emb_out)
            else:
                x = layer(x, t=t, prev_emb_out=prev_emb_out)
        return x, prev_emb_out


class BaseQuantBlock(nn.Module):
    def __init__(self, aq_params=None, **kwargs):
        super().__init__()
        self.use_wq = False
        self.use_aq = False
        self.ignore_recon = False
        self.disable_aq = False

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)


class QuantUpsample(BaseQuantBlock):
    def __init__(self, upsample, aq_params, **kwargs):
        super(QuantUpsample, self).__init__(aq_params)

        self.channels = upsample.channels
        self.out_channels = upsample.out_channels or upsample.channels
        self.use_conv = upsample.use_conv
        self.dims = upsample.dims
        if upsample.use_conv:
            self.conv = upsample.conv
    
    def forward(self, x, t=None, prev_emb_out=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            if isinstance(self.conv, QuantLayer):
                x = self.conv(x, t, prev_emb_out)
            else:
                x = self.conv(x)
        return x
    

class QuantDownsample(BaseQuantBlock):
    def __init__(self, downsample, aq_params, **kwargs):
        super(QuantDownsample, self).__init__(aq_params)

        self.channels = downsample.channels
        self.out_channels = downsample.out_channels or downsample.channels
        self.use_conv = downsample.use_conv
        self.dims = downsample.dims
        self.op = downsample.op

    def forward(self, x, t=None, prev_emb_out=None):
        if isinstance(self.op, QuantLayer):
            x = self.op(x, t, prev_emb_out)
        else:
            x = self.op(x)
        return x


class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(self, res, aq_params, **kwargs):
        super().__init__(aq_params)

        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = False # res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self, x, emb=None, t=None, prev_emb_out=None, split=0):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
                self._forward, (x, emb, t, prev_emb_out, split), self.parameters(), self.use_checkpoint
            )

    def _forward(self, x, emb=None, t=None, prev_emb_out=None, split=0):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h, t, prev_emb_out)
        else:
            h = self.in_layers[:-1](x)
            h = self.in_layers[-1](h, t, prev_emb_out)

        h_shape = [h.size(0), -1] + (len(h.size()) - 2) * [1]
        emb_out = self.emb_layers(emb).type(x.dtype)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out.reshape(*h_shape), 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest[:-1](h)
            h = out_rest[-1](h, t ,emb_out)
        else:
            h = h + emb_out.reshape(*h_shape)
            h = self.out_layers[:-1](h)
            h = self.out_layers[-1](h, t, emb_out)
        
        if isinstance(self.skip_connection, QuantLayer):
            return self.skip_connection(x, t, prev_emb_out, split) + h, emb_out

        else:
            return self.skip_connection(x) + h, emb_out
    
    def _forward_temp(self, x, emb=None, t=None, prev_emb_out=None, split=0):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers[:-1](x)
            h = self.in_layers[-1](h, t)

        h_shape = [h.size(0), -1] + (len(h.size()) - 2) * [1]
        emb_out = self.emb_layers(emb).type(x.dtype)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out.reshape(*h_shape), 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out.reshape(*h_shape)
            h = self.out_layers[:-1](h)
            h = self.out_layers[-1](h, t)
        
        if isinstance(self.skip_connection, QuantLayer):
            return self.skip_connection(x, t, None, split) + h, emb_out

        else:
            return self.skip_connection(x) + h, emb_out


class QuantQKMatMul(BaseQuantBlock):
    def __init__(self, aq_params, **kwargs):
        super().__init__(aq_params)
        self.scale = None
        self.use_aq = False
        self.aqtizer_q = ActivationQuantizer(**aq_params)
        self.aqtizer_k = ActivationQuantizer(**aq_params)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor
                ) -> torch.Tensor:
        if self.use_aq and not self.disable_aq:
            quant_q = self.aqtizer_q(q * self.scale)
            quant_k = self.aqtizer_k(k * self.scale)
            weight = torch.einsum(
                "bct,bcs->bts", quant_q, quant_k
            )
        else:
            weight = torch.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight


class QuantSMVMatMul(BaseQuantBlock):
    def __init__(self, aq_params, softmax_a_bit=8, **kwargs):
        super().__init__(aq_params)
        self.use_aq = False
        self.aqtizer_v = ActivationQuantizer(**aq_params)
        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_a_bit
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        self.aqtizer_w = ActivationQuantizer(**aq_params_w)

    def forward(self,
                weight: torch.Tensor,
                v: torch.Tensor
                ) -> torch.Tensor:
        if self.use_aq and not self.disable_aq:
            a = torch.einsum("bts,bcs->bct", self.aqtizer_w(weight), self.aqtizer_v(v))
        else:
            a = torch.einsum("bts,bcs->bct", weight, v)
        return a


class QuantAttentionBlock(BaseQuantBlock):
    def __init__(self, attn, aq_params, **kwargs):
        super().__init__(aq_params)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = False # attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv

        self.attention = attn.attention

        self.proj_out = attn.proj_out

    def forward(self, x, t=None, prev_emb_out=None):
        return checkpoint(self._forward, (x, t, prev_emb_out), self.parameters(), True)

    def _forward(self, x, t=None, prev_emb_out=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x), t, prev_emb_out)
        h = self.attention(qkv)
        h = self.proj_out(h, t, prev_emb_out)
        return (x + h).reshape(b, c, *spatial)


class QuantGEGLU(BaseQuantBlock):
    def __init__(self, geglu, aq_params):
        super().__init__(aq_params)
        self.proj = geglu.proj

    def forward(self, x, t=None, prev_emb_out=None):
        x, gate = self.proj(x, t, prev_emb_out).chunk(2, dim=-1)
        return x * F.gelu(gate)


class QuantFeedForward(BaseQuantBlock):
    def __init__(self, ff, aq_params):
        super().__init__(aq_params)

        self.net = ff.net

    def forward(self, x, t=None, prev_emb_out=None):
        x = self.net[0](x, t, prev_emb_out)
        x = self.net[1](x)
        x = self.net[2](x, t, prev_emb_out)
        return x


class QuantCrossAttention(BaseQuantBlock):
    def __init__(self, attn, aq_params, softmax_a_bit=8, **kwargs):
        super().__init__(aq_params)

        self.scale = attn.scale
        self.heads = attn.heads

        self.to_q = attn.to_q
        self.to_k = attn.to_k
        self.to_v = attn.to_v

        self.to_out = attn.to_out

        self.aqtizer_q = ActivationQuantizer(**aq_params)
        self.aqtizer_k = ActivationQuantizer(**aq_params)
        self.aqtizer_v = ActivationQuantizer(**aq_params)

        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_a_bit
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        self.aqtizer_w = ActivationQuantizer(**aq_params)

    def forward(self, x, context=None, mask=None, t=None, prev_emb_out=None):
        h = self.heads

        q = self.to_q(x, t, prev_emb_out)
        context = default(context, x)
        k = self.to_k(context, t, prev_emb_out)
        v = self.to_v(context, t, prev_emb_out)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if self.use_aq and not self.disable_aq:
            quant_q = self.aqtizer_q(q)
            quant_k = self.aqtizer_k(k)
            sim = torch.einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)

        if self.use_aq and not self.disable_aq:
            out = torch.einsum('b i j, b j d -> b i d', self.aqtizer_w(attn), self.aqtizer_v(v))
        else:
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out[0](out, t, prev_emb_out)
        out = self.to_out[-1](out)
        return out


class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(self, tran, aq_params, softmax_a_bit=8, **kwargs):
        super().__init__(aq_params)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2

        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint =  False

        self.attn1.use_aq = False
        self.attn2.use_aq = False
        self.attn1.disable_aq = False
        self.attn2.disable_aq = False

    def forward(self, x, context=None, t=None, prev_emb_out=None):
        return checkpoint(self._forward, (x, context, t, prev_emb_out), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, t=None, prev_emb_out=None):
        assert context is not None

        x = self.attn1(self.norm1(x), t=t, prev_emb_out=prev_emb_out) + x
        x = self.attn2(self.norm2(x), context, t=t, prev_emb_out=prev_emb_out) + x
        x = self.ff(self.norm3(x), t=t, prev_emb_out=prev_emb_out) + x
        return x
    

class QuantSpatialTransformer(BaseQuantBlock):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, attn, aq_params, **kwargs):
        super().__init__(aq_params)
        self.in_channels = attn.in_channels
        self.norm = attn.norm

        self.proj_in = attn.proj_in
        self.transformer_blocks = attn.transformer_blocks
        self.proj_out = attn.proj_out

    def forward(self, x, context=None, t=None, prev_emb_out=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x, t, prev_emb_out)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context, t=t, prev_emb_out=prev_emb_out)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x, t, prev_emb_out)
        return x + x_in
