import torch as th
import torch.nn as nn
import numpy as np
import math
import random
from PIL import Image

from modules.wnres import *
from modules.lvae import LadderVAE
from modules.eca_module import eca_layer
from utils.helpers import rgb_loss, get_bit_plane, get_i1, encode_with_p1, decode_with_params
from modules.guided_diffusion.unet import ResBlockWOTE, UNetModel
from utils.coder.mixcoder import MixEncoder, MixDecoder

nonlinearity = nn.SiLU

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

class SIG(nn.Module):
    def __init__(self,
                sig_num:int,
                n_layers:int,
                plane_width:int,
                plane_depth:int,
                latent_width:int,
                latent_depth:int,
                latent_channels:int,
                space_width:int,
                entropy_width:int,
                entropy_depth:int,
                dropout:float=0.,
                act=nn.SiLU,
                share=False) -> None:
        super().__init__()
        self.sig_num = sig_num
        self.lvae = LadderVAE(n_layers, 
                              3, 
                              latent_width,
                              latent_channels,
                              latent_width,
                              latent_depth,
                              act,
                              dropout
                              )
        self.plane_context = UNetModel(3,
                                     plane_width,
                                     plane_depth,
                                     (),
                                     dropout,
                                     tuple([1 for _ in range(n_layers+1)]
                            ))
        self.share = share
        if share:
            self.space_contexts = nn.ModuleList(
                [
                    nn.Sequential(
                        conv3x3(3, space_width),
                        ResBlockWOTE(space_width, dropout, space_width),
                    )
                    for _ in range(self.sig_num)
                ]
            )

            self.entropys = nn.ModuleList()
            for i in range(self.sig_num):
                width = latent_width + space_width + plane_width*int(i>0)
                self.entropys.append(
                    nn.Sequential(
                        eca_layer(width),
                        conv1x1(width, entropy_width),
                        *[
                                ResBlockWOTE(entropy_width, dropout, entropy_width)
                                for _ in range(entropy_depth)
                        ], 
                        nonlinearity(),
                        conv3x3(entropy_width, 6)
                    )
                )
        else:
            self.space_contexts = nn.ModuleList(
                [
                    nn.Sequential(
                        conv3x3(3, space_width),
                        ResBlockWOTE(space_width, dropout, space_width),
                    )
                    for _ in range(self.sig_num*3)
                ]
            )

            self.entropys = nn.ModuleList()
            for i in range(self.sig_num):
                for j in range(4):
                    width = latent_width + space_width*int(j>0) + plane_width*int(i>0)
                    self.entropys.append(
                        nn.Sequential(
                            eca_layer(width),
                            conv1x1(width, entropy_width),
                            *[
                                ResBlockWOTE(entropy_width, dropout, entropy_width)
                                for _ in range(entropy_depth)
                            ],    
                            nonlinearity(),
                            conv3x3(entropy_width, 6)
                        )
                    )

    def forward(self, x:th.Tensor, qps:list=None) -> th.Tensor:
        xsize = np.prod(x.shape)
        b = x.shape[0]
        sig_planes = (x >> (8-self.sig_num) << (8-self.sig_num)).float()/255.0
        t_base = th.ones(b, device=x.device)
        timesteps = th.cat([t_base*i for i in range(1, self.sig_num)])
        hyperprior, loss_ps, loss_qs = self.lvae(sig_planes, qps)
        hyperprior = hyperprior.repeat(4, 1, 1, 1)
        plane_contexts = self.plane_context(th.cat([(x >> (8-i) << (8-i)).float()/255.0 for i in range(1, self.sig_num)], dim=0), timesteps)
        plane_contexts = th.chunk(plane_contexts, self.sig_num-1, dim=0)
        loss_px = th.zeros(12*self.sig_num, device=x.device)
        if self.share:
            for i in range(self.sig_num):
                xt = get_bit_plane(x, i+1)
                space_prior = th.zeros_like(xt)

                xin = space_prior.clone()
                target = xt[..., 0::2, 0::2]
                targets = [target.clone()]

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                xin = th.cat([xin, space_prior.clone()], dim=0)
                targets.append(target.clone())

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                xin = th.cat([xin, space_prior.clone()], dim=0)
                targets.append(target.clone())

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                xin = th.cat([xin, space_prior.clone()], dim=0)
                targets.append(target.clone())
                if i == 0:
                    params = self.entropys[i](th.cat([hyperprior, self.space_contexts[i](xin)], dim=1))
                else:
                    params = self.entropys[i](th.cat([plane_contexts[i-1].repeat(4,1,1,1), hyperprior, self.space_contexts[i](xin)], dim=1))
                params = th.chunk(params, 4, dim=0)
                loss_px[12*i:12*i+3] = rgb_loss(params[0][...,0::2,0::2], targets[0])
                loss_px[12*i+3:12*i+6] = rgb_loss(params[1][...,1::2,1::2], targets[1])
                loss_px[12*i+6:12*i+9] = rgb_loss(params[2][...,0::2,1::2], targets[2])
                loss_px[12*i+9:12*i+12] = rgb_loss(params[3][...,1::2,0::2], targets[3])
        else:
            for i in range(self.sig_num):
                if i > 0:
                    plane_context = plane_contexts[i-1]
                xt = get_bit_plane(x, i+1)
                space_prior = th.zeros_like(xt)

                target = xt[..., 0::2, 0::2]
                prior =hyperprior if i ==0 else th.cat([plane_context, hyperprior], dim=1)
                params = self.entropys[4*i](prior)[...,0::2,0::2]
                loss_px[12*i:12*i+3] = rgb_loss(params, target)

                target = xt[..., 1::2, 1::2]
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                space_context = self.space_contexts[3*i](space_prior.clone())
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+1](prior)[...,1::2,1::2]
                loss_px[12*i+3:12*i+6] = rgb_loss(params, target)

                target = xt[..., 0::2, 1::2]
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                space_context = self.space_contexts[3*i+1](space_prior.clone())
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+2](prior)[...,0::2,1::2]
                loss_px[12*i+6:12*i+9] = rgb_loss(params, target)

                target = xt[..., 1::2, 0::2]
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                space_context = self.space_contexts[3*i+2](space_prior.clone())
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+3](prior)[...,1::2,0::2]
                loss_px[12*i+9:12*i+12] = rgb_loss(params, target)
        return th.cat([loss_ps, loss_qs, loss_px/xsize], dim=0)

    def inference(self, x, qps:list=None):
        xsize = np.prod(x.shape)
        b = x.shape[0]
        sig_planes = (x >> (8-self.sig_num) << (8-self.sig_num)).float()/255.0
        t_base = th.ones(b, device=x.device)
        hyperprior, loss_ps, loss_qs = self.lvae(sig_planes, qps)

        loss_px = th.zeros(12*self.sig_num, device=x.device)
        if self.share:
            for i in range(self.sig_num):
                idx = 8-i
                previous_planes = (x >> idx << idx).float()/255.0
                if i > 0:
                    plane_context = self.plane_context(previous_planes, t_base * i)
                xt = get_bit_plane(x, i+1)
                space_prior = th.zeros_like(xt)
                
                target = xt[..., 0::2, 0::2]
                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = self.entropys[i](prior)[...,0::2,0::2]
                loss_px[12*i:12*i+3] = rgb_loss(params, target)

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = self.entropys[i](prior)[...,1::2,1::2]
                loss_px[12*i+3:12*i+6] = rgb_loss(params, target)

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = self.entropys[i](prior)[...,0::2,1::2]
                loss_px[12*i+6:12*i+9] = rgb_loss(params, target)

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = self.entropys[i](prior)[...,1::2,0::2]
                loss_px[12*i+9:12*i+12] = rgb_loss(params, target)
        else:
            for i in range(self.sig_num):
                idx = 8-i
                previous_planes = (x >> idx << idx).float()/255.0
                if i > 0:
                    plane_context = self.plane_context(previous_planes, t_base * i)
                xt = get_bit_plane(x, i+1)
                space_prior = th.zeros_like(xt)
                
                target = xt[..., 0::2, 0::2]
                prior = hyperprior if i ==0 else th.cat([plane_context, hyperprior], dim=1)
                params = self.entropys[4*i](prior)[...,0::2,0::2]
                loss_px[12*i:12*i+3] = rgb_loss(params, target.clone())

                target = xt[..., 1::2, 1::2]
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                space_context = self.space_contexts[3*i](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+1](prior)[...,1::2,1::2]
                loss_px[12*i+3:12*i+6] = rgb_loss(params, target.clone())

                target = xt[..., 0::2, 1::2]
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                space_context = self.space_contexts[3*i+1](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+2](prior)[...,0::2,1::2]
                loss_px[12*i+6:12*i+9] = rgb_loss(params, target.clone())

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                space_context = self.space_contexts[3*i+2](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+3](prior)[...,1::2,0::2]
                loss_px[12*i+9:12*i+12] = rgb_loss(params, target.clone())

        return th.cat([loss_ps, loss_qs, loss_px/xsize], dim=0)
    
    def _encode_single(self, x:th.Tensor, coder:MixEncoder):
        # decode qz
        sig_planes = (x >> (8-self.sig_num) << (8-self.sig_num)).float()/255.0
        zs, hyperprior = self.lvae._decompress_qz(sig_planes, coder)
        codeLength_mid = coder.getStreamLength()

        # encode px
        t_base = th.ones(1, device=x.device)
        if self.share:
            for i in reversed(range(self.sig_num)):
                idx = 8-i
                previous_planes = (x >> idx << idx).float()/255.0
                if i > 0:
                    plane_context = self.plane_context(previous_planes, t_base * i)
                xt = get_bit_plane(x, i+1)
                space_prior = th.zeros_like(xt)

                target = xt[..., 0::2, 0::2]
                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = [self.entropys[i](prior)[...,0::2,0::2]]
                targets = [target.clone()]

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params.append(self.entropys[i](prior)[...,1::2,1::2])
                targets.append(target.clone())

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params.append(self.entropys[i](prior)[...,0::2,1::2])
                targets.append(target.clone())

                target = xt[..., 1::2, 0::2]
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params.append(self.entropys[i](prior)[...,1::2,0::2])
                targets.append(target.clone())

                i1 = th.cat([
                    get_i1(params[0], targets[0]),
                    get_i1(params[1], targets[1]),
                    get_i1(params[2], targets[2]),
                    get_i1(params[3], targets[3])
                ], dim=0)
                bins = th.cat(targets, dim=0)
                encode_with_p1(i1, bins, coder)
        else:
            for i in reversed(range(self.sig_num)):
                idx = 8-i
                previous_planes = (x >> idx << idx).float()/255.0
                if i > 0:
                    plane_context = self.plane_context(previous_planes, t_base * i)
                xt = get_bit_plane(x, i+1)
                space_prior = th.zeros_like(xt)

                target = xt[..., 0::2, 0::2]
                prior = hyperprior if i ==0 else th.cat([plane_context, hyperprior], dim=1)
                params = [self.entropys[4*i](prior)[...,0::2,0::2]]
                targets = [target.clone()]

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                space_context = self.space_contexts[3*i](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params.append(self.entropys[4*i+1](prior)[...,1::2,1::2])
                targets.append(target.clone())

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                space_context = self.space_contexts[3*i+1](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params.append(self.entropys[4*i+2](prior)[...,0::2,1::2])
                targets.append(target.clone())

                target = xt[..., 1::2, 0::2]
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                space_context = self.space_contexts[3*i+2](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params.append(self.entropys[4*i+3](prior)[...,1::2,0::2])
                targets.append(target.clone())

                i1 = th.cat([
                    get_i1(params[0], targets[0]),
                    get_i1(params[1], targets[1]),
                    get_i1(params[2], targets[2]),
                    get_i1(params[3], targets[3])
                ], dim=0)
                bins = th.cat(targets, dim=0)
                encode_with_p1(i1, bins, coder)
        
        # encode pz
        zs_r = zs[::-1].copy().astype(np.uint16)
        coder.encodeUniforms(zs_r)
        return codeLength_mid

    def _decode_single(self, h:int, w:int, device:str, coder:MixDecoder):
        t_base = th.ones(1, device=device)
        xt = th.zeros((1,3,h,w), device=device).float()
        # decode qz
        mups, inputs, z_ids, hyperprior = self.lvae._decompress_pz(h, w, coder, device)
        # decode px
        if self.share:
            for i in range(self.sig_num):
                previous_planes = xt
                if i > 0:
                    plane_context = self.plane_context(previous_planes, t_base * i)
                space_prior = th.zeros_like(xt)

                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i ==0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = self.entropys[i](prior)[...,0::2,0::2]
                space_prior[..., 0::2, 0::2] = decode_with_params(params, coder) * 2 - 1

                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i == 0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = self.entropys[i](prior)[...,1::2,1::2]
                space_prior[..., 1::2, 1::2] = decode_with_params(params, coder) * 2 - 1

                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i == 0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = self.entropys[i](prior)[...,0::2,1::2]
                space_prior[..., 0::2, 1::2] = decode_with_params(params, coder) * 2 - 1

                prior = th.cat([hyperprior, self.space_contexts[i](space_prior)], dim=1) if i == 0 else th.cat([plane_context, hyperprior, self.space_contexts[i](space_prior)], dim=1)
                params = self.entropys[i](prior)[...,1::2,0::2]
                space_prior[..., 1::2, 0::2] = decode_with_params(params, coder) * 2 - 1

                xi = (space_prior + 1) / 2
                xt += xi * (1 << (7-i))/255.0
        else:
            for i in range(self.sig_num):
                previous_planes = xt
                if i > 0:
                    plane_context = self.plane_context(previous_planes, t_base * i)
                space_prior = th.zeros_like(xt)

                prior = hyperprior if i ==0 else th.cat([plane_context, hyperprior], dim=1)
                params = self.entropys[4*i](prior)[...,0::2,0::2]
                space_prior[..., 0::2, 0::2] = decode_with_params(params, coder) * 2 - 1

                space_context = self.space_contexts[3*i](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+1](prior)[...,1::2,1::2]
                space_prior[..., 1::2, 1::2] = decode_with_params(params, coder) * 2 - 1

                space_context = self.space_contexts[3*i+1](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+2](prior)[...,0::2,1::2]
                space_prior[..., 0::2, 1::2] = decode_with_params(params, coder) * 2 - 1

                space_context = self.space_contexts[3*i+2](space_prior)
                prior = th.cat([hyperprior, space_context], dim=1) if i ==0 else th.cat([plane_context, hyperprior, space_context], dim=1)
                params = self.entropys[4*i+3](prior)[...,1::2,0::2]
                space_prior[..., 1::2, 0::2] = decode_with_params(params, coder) * 2 - 1

                xi = (space_prior + 1) / 2
                xt += xi * (1 << (7-i))/255.0
        
        self.lvae._compress_qz(xt, z_ids, mups, inputs, coder)
        return xt*255.0

class INS(nn.Module):
    def __init__(self,
                sig_num:int,
                n_layers:int,
                plane_width:int,
                plane_depth:int,
                space_width:int,
                entropy_width:int,
                entropy_depth:int,
                dropout:float=0.,
                share:float=False):
        super().__init__()
        self.ins_num = 8 - sig_num
        self.plane_context = UNetModel(3,
                                     plane_width,
                                     plane_depth,
                                     (),
                                     dropout,
                                     tuple([1 for _ in range(n_layers)])
                                    )
        self.share = share
        if share:
            self.space_contexts = nn.ModuleList(
                [
                    nn.Sequential(
                        conv3x3(3, space_width),
                        ResBlockWOTE(space_width, dropout, space_width),
                    )
                    for _ in range(self.ins_num)
                ]
            )

            self.entropys = nn.ModuleList(
                [
                    nn.Sequential(
                        eca_layer(space_width+plane_width),
                        conv1x1(space_width+plane_width, entropy_width),
                        *[
                            ResBlockWOTE(entropy_width, dropout, entropy_width)
                            for _ in range(entropy_depth)
                        ], 
                        nonlinearity(),
                        conv3x3(entropy_width, 6)
                    )
                    for _ in range(self.ins_num)
                ]
            )
        else:
            self.space_contexts = nn.ModuleList(
                [
                    nn.Sequential(
                        conv3x3(3, space_width),
                        ResBlockWOTE(space_width, dropout, space_width),
                    )
                    for _ in range(self.ins_num*3)
                ]
            )

            self.entropys = nn.ModuleList()
            for _ in range(self.ins_num):
                for j in range(4):
                    width = space_width*int(j>0) + plane_width
                    self.entropys.append(
                        nn.Sequential(
                            eca_layer(width),
                            conv1x1(width, entropy_width),
                            *[
                                ResBlockWOTE(entropy_width, dropout, entropy_width)
                                for _ in range(entropy_depth)
                            ],    
                            nonlinearity(),
                            conv3x3(entropy_width, 6)
                        )
                    )
        
    def forward(self, x:th.Tensor):
        xsize = np.prod(x.shape)
        b = x.shape[0]
        bpds = th.zeros(12*self.ins_num, device=x.device)
        t_base = th.ones(b, device=x.device)
        timesteps = th.cat([t_base*i for i in range(1,self.ins_num+1)])
        plane_contexts = self.plane_context(th.cat([
            (x >> self.ins_num-i << self.ins_num-i).float()/255.0 for i in range(self.ins_num)
        ], dim=0), timesteps)
        plane_contexts = th.chunk(plane_contexts, self.ins_num, dim=0)

        if self.share:
            for i in range(self.ins_num):
                xt = get_bit_plane(x, i+self.ins_num+1)
                space_prior = th.zeros_like(xt)

                xin = space_prior.clone()
                target = xt[..., 0::2, 0::2]
                targets = [target.clone()]

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                xin = th.cat([xin, space_prior.clone()], dim=0)
                targets.append(target.clone())

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                xin = th.cat([xin, space_prior.clone()], dim=0)
                targets.append(target.clone())

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                xin = th.cat([xin, space_prior.clone()], dim=0)
                targets.append(target.clone())

                params = self.entropys[i](th.cat([plane_contexts[i].repeat(4, 1, 1, 1), self.space_contexts[i](xin)], dim=1))
                params = th.chunk(params, 4, dim=0)
                bpds[12*i:12*i+3] = rgb_loss(params[0][...,0::2,0::2], targets[0])
                bpds[12*i+3:12*i+6] = rgb_loss(params[1][...,1::2,1::2], targets[1])
                bpds[12*i+6:12*i+9] = rgb_loss(params[2][...,0::2,1::2], targets[2])
                bpds[12*i+9:12*i+12] = rgb_loss(params[3][...,1::2,0::2], targets[3])
        else:
            for i in range(self.ins_num):
                plane_context = plane_contexts[i]
                xt = get_bit_plane(x, i+self.ins_num+1)
                space_prior = th.zeros_like(xt)

                target = xt[..., 0::2, 0::2]
                prior = plane_context
                params = self.entropys[4*i](prior)[...,0::2,0::2]
                bpds[i*12:i*12+3] = rgb_loss(params, target.clone())

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                space_context = self.space_contexts[3*i](space_prior.clone())
                prior = th.cat([plane_context, space_context], dim=1)
                params = self.entropys[4*i+1](prior)[...,1::2,1::2]
                bpds[i*12+3:i*12+6] = rgb_loss(params, target.clone())

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                space_context = self.space_contexts[3*i+1](space_prior.clone())
                prior = th.cat([plane_context, space_context], dim=1)
                params = self.entropys[4*i+2](prior)[...,0::2,1::2]
                bpds[i*12+6:i*12+9] = rgb_loss(params, target.clone())

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                space_context = self.space_contexts[3*i+2](space_prior.clone())
                prior = th.cat([plane_context, space_context], dim=1)
                params = self.entropys[4*i+3](prior)[...,1::2,0::2]
                bpds[i*12+9:i*12+12] = rgb_loss(params, target.clone())
        return bpds/xsize

    def inference(self, x:th.Tensor):
        xsize = np.prod(x.shape)
        b = x.shape[0]
        bpds = th.zeros(12*self.ins_num, device=x.device)
        t_base = th.ones(b, device=x.device)

        if self.share:
            for i in range(self.ins_num):
                idx = self.ins_num-i
                previous_planes = (x >> idx << idx).float()/255.0
                plane_context = self.plane_context(previous_planes, t_base * (i+1))
                xt = get_bit_plane(x, i+self.ins_num+1)
                space_prior = th.zeros_like(xt)

                target = xt[..., 0::2, 0::2]
                params = self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))
                bpds[i*12:i*12+3] = rgb_loss(params[...,0::2,0::2], target)

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                params = self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))
                bpds[i*12+3:i*12+6] = rgb_loss(params[...,1::2,1::2], target)

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                params = self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))
                bpds[i*12+6:i*12+9] = rgb_loss(params[...,0::2,1::2], target)

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                params = self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))
                bpds[i*12+9:i*12+12] = rgb_loss(params[...,1::2,0::2], target)
        else:
            for i in range(self.ins_num):
                idx = self.ins_num-i
                previous_planes = (x >> idx << idx).float()/255.0
                plane_context = self.plane_context(previous_planes, t_base * (i+1))
                xt = get_bit_plane(x, i+self.ins_num+1)
                space_prior = th.zeros_like(xt)

                target = xt[..., 0::2, 0::2]
                params = self.entropys[4*i](plane_context)
                bpds[i*12:i*12+3] = rgb_loss(params[...,0::2,0::2], target)

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                params = self.entropys[4*i+1](th.cat([plane_context, self.space_contexts[3*i](space_prior)], dim=1))
                bpds[i*12+3:i*12+6] = rgb_loss(params[...,1::2,1::2], target)

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                params = self.entropys[4*i+2](th.cat([plane_context, self.space_contexts[3*i+1](space_prior)], dim=1))
                bpds[i*12+6:i*12+9] = rgb_loss(params[...,0::2,1::2], target)

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                params = self.entropys[4*i+3](th.cat([plane_context, self.space_contexts[3*i+2](space_prior)], dim=1))
                bpds[i*12+9:i*12+12] = rgb_loss(params[...,1::2,0::2], target)
        
        return bpds/xsize

    def _encode_single(self, x:th.Tensor, coder:MixEncoder):
        t_base = th.ones(1, device=x.device)
        
        if self.share:
            for i in reversed(range(self.ins_num)):
                idx = self.ins_num-i
                previous_planes = (x >> idx << idx).float()/255.0
                xt = get_bit_plane(x, i+self.ins_num+1)
                plane_context = self.plane_context(previous_planes, t_base * (i+1))

                target = xt[..., 0::2, 0::2]
                space_prior = th.zeros_like(xt)
                params = [self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))[...,0::2,0::2]]
                targets = [target.clone()]

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                params.append(self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))[...,1::2,1::2])
                targets.append(target.clone())

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                params.append(self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))[...,0::2,1::2])
                targets.append(target.clone())

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                params.append(self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))[...,1::2,0::2])
                targets.append(target.clone())

                i1 = th.cat([
                    get_i1(params[0], targets[0]),
                    get_i1(params[1], targets[1]),
                    get_i1(params[2], targets[2]),
                    get_i1(params[3], targets[3])
                ], dim=0)
                bins = th.cat(targets, dim=0)
                encode_with_p1(i1, bins, coder)
        else:
            for i in reversed(range(self.ins_num)):
                idx = self.ins_num-i
                previous_planes = (x >> idx << idx).float()/255.0
                xt = get_bit_plane(x, i+self.ins_num+1)
                plane_context = self.plane_context(previous_planes, t_base * (i+1))
                space_prior = th.zeros_like(xt)

                target = xt[..., 0::2, 0::2]
                params = [self.entropys[4*i](plane_context)[...,0::2,0::2]]
                targets = [target.clone()]

                target = xt[..., 1::2, 1::2] 
                space_prior[..., 0::2, 0::2] = xt[..., 0::2, 0::2] * 2 - 1
                params.append(self.entropys[4*i+1](th.cat([plane_context, self.space_contexts[3*i](space_prior)], dim=1))[...,1::2,1::2])
                targets.append(target.clone())

                target = xt[..., 0::2, 1::2] 
                space_prior[..., 1::2, 1::2] = xt[..., 1::2, 1::2] * 2 - 1
                params.append(self.entropys[4*i+2](th.cat([plane_context, self.space_contexts[3*i+1](space_prior)], dim=1))[...,0::2,1::2])
                targets.append(target.clone())

                target = xt[..., 1::2, 0::2] 
                space_prior[..., 0::2, 1::2] = xt[..., 0::2, 1::2] * 2 - 1
                params.append(self.entropys[4*i+3](th.cat([plane_context, self.space_contexts[3*i+2](space_prior)], dim=1))[...,1::2,0::2])
                targets.append(target.clone())

                i1 = th.cat([
                    get_i1(params[0], targets[0]),
                    get_i1(params[1], targets[1]),
                    get_i1(params[2], targets[2]),
                    get_i1(params[3], targets[3])
                ], dim=0)
                bins = th.cat(targets, dim=0)
                encode_with_p1(i1, bins, coder)
        codeLength = coder.getStreamLength()
        return codeLength

    def _decode_single(self, xt:th.Tensor, device:str, coder:MixDecoder):
        t_base = th.ones(1, device=device)
        if self.share:
            for i in range(self.ins_num):
                plane_context = self.plane_context(xt, t_base * (i+1))
                space_prior = th.zeros_like(xt)
                params = self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))[...,0::2,0::2]
                space_prior[..., 0::2, 0::2] = decode_with_params(params, coder) * 2 - 1
                params = self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))[...,1::2,1::2]
                space_prior[..., 1::2, 1::2] = decode_with_params(params, coder) * 2 - 1

                params = self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))[...,0::2,1::2]
                space_prior[..., 0::2, 1::2] = decode_with_params(params, coder) * 2 - 1

                params = self.entropys[i](th.cat([plane_context, self.space_contexts[i](space_prior)], dim=1))[...,1::2,0::2]
                space_prior[..., 1::2, 0::2] = decode_with_params(params, coder) * 2 - 1

                xi = (space_prior + 1) / 2
                xt += xi * (1 << (self.ins_num-i-1))/255.0
        else:
            for i in range(self.ins_num):
                plane_context = self.plane_context(xt, t_base * (i+1))
                space_prior = th.zeros_like(xt)
                params = self.entropys[4*i](plane_context)[...,0::2,0::2]
                space_prior[..., 0::2, 0::2] = decode_with_params(params, coder) * 2 - 1
                
                params = self.entropys[4*i+1](th.cat([plane_context, self.space_contexts[3*i](space_prior)], dim=1))[...,1::2,1::2]
                space_prior[..., 1::2, 1::2] = decode_with_params(params, coder) * 2 - 1

                params = self.entropys[4*i+2](th.cat([plane_context, self.space_contexts[3*i+1](space_prior)], dim=1))[...,0::2,1::2]
                space_prior[..., 0::2, 1::2] = decode_with_params(params, coder) * 2 - 1

                params = self.entropys[4*i+3](th.cat([plane_context, self.space_contexts[3*i+2](space_prior)], dim=1))[...,1::2,0::2]
                space_prior[..., 1::2, 0::2] = decode_with_params(params, coder) * 2 - 1

                xi = (space_prior + 1) / 2
                xt += xi * (1 << (self.ins_num-i-1))/255.0
        return xt*255.0
    
class ARIB_BPS(nn.Module):
    def __init__(self,
                sig_num:int,
                n_layers:int,
                sig_plane_width:int,
                sig_plane_depth:int,
                ins_plane_width:int,
                ins_plane_depth:int,
                latent_width:int,
                latent_depth:int,
                latent_channels:int,
                space_width:int,
                entropy_width:int,
                entropy_depth:int,
                dropout:float=0.,
                act=nn.SiLU,
                space_width_ins=None,
                entropy_width_ins=None,
                entropy_depth_ins=None,
                share:bool=False,
                ) -> None:
        super().__init__()
        self.sig = SIG(
            sig_num,
            n_layers,
            sig_plane_width,
            sig_plane_depth,
            latent_width,
            latent_depth,
            latent_channels,
            space_width,
            entropy_width,
            entropy_depth,
            dropout,
            act,
            share
        )
        if space_width_ins is None:
            self.ins = INS(
                sig_num,
                n_layers+1,
                ins_plane_width,
                ins_plane_depth,
                space_width,
                entropy_width,
                entropy_depth,
                dropout,
                share
            )
        else:
            self.ins = INS(
                sig_num,
                n_layers+1,
                ins_plane_width,
                ins_plane_depth,
                space_width_ins,
                entropy_width_ins,
                entropy_depth_ins,
                dropout,
                share
            )
        self.n_layers = n_layers
        self.z_channels = latent_channels
        self.ql = 3
        self.qu = 10
        self.interval = self.qu - self.ql +  1
        self.precision = 16
        self.scale = 1 << self.precision
    
    def load(self, sigpath:str, inspath:str):
        self.sig.load_state_dict(th.load(sigpath, map_location='cpu'))
        self.ins.load_state_dict(th.load(inspath, map_location='cpu'))

    def forward(self,x:th.Tensor, qps:list=None):
        return th.cat([self.sig(x, qps), self.ins(x)], dim=0)

    def inference(self, x:th.Tensor, qps:list=None):
        return th.cat([self.sig.inference(x, qps), self.ins.inference(x)], dim=0)
    
    def _encode_dataset(self, x:th.Tensor, coder:MixEncoder, isFirst=True):
        self.ins._encode_single(x, coder)
        isPop = None
        if isFirst:
            isPop = coder.switchToRans()
        self.sig._encode_single(x, coder)
        return isPop
    
    def _decode_dataset(self, h:int, w:int, device:str, coder:MixDecoder, isPop=None):
        xt = self.sig._decode_single(h, w, device, coder)
        if isPop is not None:
            coder.switchFromRans(isPop)
        return self.ins._decode_single(th.round(xt)/255.0, device, coder)
    
    def _encode_single(self, x:th.Tensor, coder:MixEncoder, precision=16):
        xsize = x.shape[2] * x.shape[3]
        zsize = 0.25*((1-0.25**self.n_layers)/0.75)*xsize*self.z_channels
        self.ins._encode_single(x, coder)
        init_bits = coder.getInitSize()
        qp = int(init_bits/zsize)
        if qp < self.ql:
            pad_num = math.ceil((self.ql-qp)*zsize/64)
            qp = self.ql
            success = False
            while success == False:
                init_stream = np.array([random.randint(0, (1<<64)-1) for _ in range(pad_num)], dtype=np.uint64)
                coder = MixEncoder(precision, qp, init_stream, xsize//4)
                try:
                    self.ins._encode_single(x, coder)
                    self.sig.lvae.set_qp(qp, x.device)
                    isPop = coder.switchToRans()
                    coder.setPrecision(qp)
                    self.sig._encode_single(x, coder)
                    success = True
                except:
                    pad_num += 1
                init_bpd = pad_num*64/np.prod(x.shape)
        else:
            qp = min(max(self.ql,qp), self.qu)
            qp = int(math.floor(qp))
            self.sig.lvae.set_qp(qp, x.device)
            isPop = coder.switchToRans()
            coder.setPrecision(qp)
            self.sig._encode_single(x, coder)
            init_bpd = 0.
        coder.setPrecision(3)
        coder.encodeUniforms(np.array([qp-self.ql]))
        return coder, qp, isPop, init_bpd
    
    def _decode_single(self, h:int, w:int, device:str, isPop:bool, coder:MixDecoder):
        coder.setPrecision(3)
        qp = coder.decodeUniforms(1)[0]+self.ql
        self.sig.lvae.set_qp(qp, device)
        coder.setPrecision(qp)
        xt = self.sig._decode_single(h, w, device, coder)
        coder.switchFromRans(isPop)
        return self.ins._decode_single(th.round(xt)/255.0, device, coder)
    
    def eval_single(self, x:th.Tensor, precision=16):
        assert x.shape[0] == 1 and x.shape[1] == 3 and x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0
        size = np.prod(x.shape)
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        encoder = MixEncoder(precision, 3, size)
        start.record()
        encoder, qp, isPop, init_bpd = self._encode_single(x, encoder)
        end.record()
        th.cuda.synchronize()
        encode_time = start.elapsed_time(end)

        codeLength = encoder.getStreamLength()
        isNotAligned = encoder.flushState()
        isLargerThan64 = encoder.flushTip()

        encoded = encoder.getStream()
        decoder = MixDecoder(precision, qp, encoded)
        decoder.prepareForDecode(isNotAligned, isLargerThan64)
        start.record()
        x_rec = self._decode_single(x.shape[2], x.shape[3], x.device, isPop, decoder)
        end.record()
        th.cuda.synchronize()
        decode_time = start.elapsed_time(end)

        x_rec = th.round(x_rec).clamp(0, 255).to(th.uint8)
        errsum = (x_rec != x).sum()

        return f'{np.prod(x.shape)},{codeLength},{encode_time:.1f},{decode_time:.1f},{int(qp)},{init_bpd:.2f},{errsum.item()}'

    def eval_dataset(self, x:th.Tensor, precision=16, qp=10):
        width = x[0].shape[2]
        num_init = int(0.25*((1-0.25**self.n_layers)/0.75)* width * width *self.z_channels*(self.qu+4))//64 + 1
        self.sig.lvae.set_qp(qp,x[0].device)

        b = len(x)
        size = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        init_stream = np.array([random.randint(0, (1<<64)-1) for _ in range(num_init)]).astype(np.uint64)
        encoder = MixEncoder(precision, qp, init_stream, width*width*b*3//4)
        start.record()
        isPop = self._encode_dataset(x[0], encoder, True)
        size += np.prod(x[0].shape)
        for i in range(b-1):
            self._encode_dataset(x[i+1], encoder, False)
            size += np.prod(x[i+1].shape)
        end.record()
        torch.cuda.synchronize()
        encode_time = start.elapsed_time(end)

        codeLength = encoder.getStreamLength()
        isNotAligned = encoder.flushState()
        isLargerThan64 = encoder.flushTip()

        encoded = encoder.getStream()
        decoder = MixDecoder(precision, qp, encoded)
        decoder.prepareForDecode(isNotAligned, isLargerThan64)
        x_rec = []
        start.record()
        for i in reversed(range(b-1)):
            x_rec.append(self._decode_dataset(x[i+1].shape[2], x[i+1].shape[3], x[i].device, decoder))
        x_rec.append(self._decode_dataset(x[0].shape[2], x[0].shape[3], x[0].device, decoder, isPop))
        
        end.record()
        torch.cuda.synchronize()
        decode_time = start.elapsed_time(end)
        err_sum = 0
        for i in range(b):
            x_rec[b-1-i] = th.round(x_rec[b-1-i]).clamp(0, 255).to(th.uint8)
            err_sum += (x_rec[b-1-i] != x[i]).sum().item()
        return f'{size},{codeLength},{encode_time:.1f},{decode_time:.1f},{err_sum}'
    
    def compress_to_file(self, x:th.Tensor, path, precision=16):
        size = np.prod(x.shape)
        encoder = MixEncoder(precision, 3, size)
        encoder, __cached__, isPop, _ = self._encode_single(x, encoder)

        isNotAligned = encoder.flushState()
        isLargerThan64 = encoder.flushTip()
        encoded = encoder.getStream()
        flags = isPop*4+isNotAligned*2+isLargerThan64
        header = np.array([x.shape[2], x.shape[3], flags], dtype=np.uint16)
        with open(path, 'wb') as f:
            f.write(header.tobytes())
            f.write(encoded.tobytes())

    def decompress_from_file(self, src, dst, device:str='cuda:0', precision=16):
        with open(src, 'rb') as f:
            header = np.frombuffer(f.read(6), dtype=np.uint16)
            flags = header[2]
            isPop = flags//4
            isNotAligned = (flags%4)//2
            isLargerThan64 = flags%2
            encoded = np.frombuffer(f.read(), dtype=np.uint64)
            decoder = MixDecoder(precision, 0, encoded)
            decoder.prepareForDecode(isNotAligned, isLargerThan64)
            x_rec = self._decode_single(int(header[0]), int(header[1]), device, isPop, decoder)
            x_rec = th.round(x_rec).clamp(0, 255).to(th.uint8)
        x_img_np = x_rec.squeeze(0).permute(1,2,0).cpu().numpy()
        Image.fromarray(x_img_np).save(dst)



        