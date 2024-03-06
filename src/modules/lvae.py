import torch as th
import torch.nn as nn
import numpy as np

from modules.wnres import *
from utils import rand
from utils.helpers import logistic_logp, logistic_eps, fake_sample
from utils.coder.mixcoder import MixEncoder, MixDecoder

class TopDownLayerBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deterministic_block:nn.Module = None
        self.stochastic:nn.Module = None

    def inference(self, 
                  input_:th.Tensor, 
                  bu_value:th.Tensor, 
                  is_top_layer:bool=False):
        if is_top_layer:
            return self.stochastic(bu_value)
        else:
            return self.stochastic(input_+bu_value)

    def generative(self, input_:th.Tensor):
        return self.stochastic(input_)

    def forward(self, 
                input_:th.Tensor,
                bu_value:th.Tensor,
                is_top_layer:bool=False,
                qps:list=None):
        # compute prior, posterior, corresponding nll and next layer input
        muq = self.inference(input_, bu_value, is_top_layer)
        if is_top_layer:
            mup = th.zeros_like(muq)
        else:
            mup = self.generative(input_)
        if qps is None:
            z = logistic_eps(muq.shape, muq.device) + muq
            log_qz = th.sum(logistic_logp(muq, z), dim=(1,2,3))/np.log(2)
            log_pz = -th.sum(logistic_logp(mup, z), dim=(1,2,3))/np.log(2)
        else:
            assert len(qps) == mup.shape[0]
            b = mup.shape[0]
            z = th.zeros_like(muq)
            log_qz = th.zeros(b, device=muq.device)
            log_pz = th.zeros(b, device=muq.device)
            for i in range(b):
                z[i,...], log_pz[i], log_qz[i] = fake_sample(mup[i,...], muq[i,...], qps[i])
            log_qz = log_qz.mean()
            log_pz = log_pz.mean()
        return self.deterministic_block(input_, z, is_top_layer), log_pz, log_qz

class TopDownDeterministicBlock(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 z_channels:int, 
                 width:int, 
                 depth:int, 
                 act = nn.PReLU, 
                 dropout:float=0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            WnRes(z_channels, in_channels, width, depth, act=act, dropout_p=dropout),
        )
        self.up = nn.Sequential(
            WnConv2d(in_channels, in_channels*4, 3, 1, 1),
            UnSqueeze2d(),
        )
    
    def forward(self, 
                input_:th.Tensor, 
                z:th.Tensor, 
                istop:bool=False):
        if istop:
            return self.up(self.net(z))
        else:
            return self.up(self.net(z)+input_)

class TopDownLayer(TopDownLayerBase):
    def __init__(self, 
                 in_channels:int,
                 z_channels:int, 
                 width:int, 
                 depth:int, 
                 act = nn.PReLU, 
                 dropout:float=0.) -> None:
        super().__init__()
        self.deterministic_block = TopDownDeterministicBlock(in_channels, z_channels, width, depth, act=act, dropout=dropout)
        self.stochastic = WnRes(in_channels, z_channels, width, depth, act=act, dropout_p=dropout)

class LadderVAEBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head:nn.Module = None
        self.top_down_layer:nn.Module = None
        self.bottom_up_layer:nn.Module = None
        self.nlayers:int = None
        self.z_channels:int = None
        self.mu_centers:np.ndarray = None
        self.z_center:np.ndarray = None

    def forward(self, x:th.Tensor, qps:list=None):
        size = np.prod(x.shape[1:])
        bu_values = []
        x = self.head(x)
        for i in range(self.nlayers):
            x = self.bottom_up_layer(x)
            bu_values.append(x)
        
        input_ = None
        loss_ps = th.zeros(self.nlayers, device=x.device)
        loss_qs = th.zeros(self.nlayers, device=x.device)
        for i in reversed(range(self.nlayers)):
            input_, log_pz, log_qz = self.top_down_layer(input_, bu_values[i], i == self.nlayers-1, qps)
            loss_ps[i] = th.mean(log_pz)/size
            loss_qs[i] = th.mean(log_qz)/size
        return input_, loss_ps, loss_qs

class LadderVAE(LadderVAEBase):
    def __init__(self, 
                 nlayers: int, 
                 in_channels:int, 
                 hidden_channels:int, 
                 z_channels:int, 
                 width:int, 
                 depth:int, 
                 act = nn.PReLU, 
                 dropout:float=0.) -> None:
        super().__init__()
        self.nlayers = nlayers
        self.top_down_layer = TopDownLayer(hidden_channels, z_channels, width, depth, act=act, dropout=dropout)
        self.head = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.bottom_up_layer = nn.Sequential(Squeeze2d(),
                                act(),
                                WnRes(hidden_channels*4, hidden_channels, width, depth, act=act, dropout_p=dropout))
        self.z_channels = z_channels
        self.mu_dir = None
        self.z_precision = 8
        self.precision = 16
        self.zcenter = None
        self.z_center = None
        self.scale = 1<<self.precision
    
    def cdfz(self, mus:np.ndarray):
        zsize = mus.shape[0]
        cdfs = np.zeros((zsize, (1<<self.z_precision)+1))
        cdfs[:,1:] = 1.0/(1+np.exp(mus[:, np.newaxis]-self.z_center))
        cdfs[:,-1] = 1
        cdfs = (cdfs*(self.scale)).astype(int)
        return cdfs
    
    def set_qp(self,qp,device):
        self.z_precision = qp
        self.zbins = rand.Bins(th.zeros(1), th.ones(1),self.z_precision)
        self.zcenter = self.zbins.centres().to(device)
        self.z_center = self.zcenter.cpu().numpy()[0]

    def _decompress_qz(self, x:th.Tensor, coder:MixEncoder):
        bu_values = []
        zs = None
        x = self.head(x)
        for i in range(self.nlayers):
            x = self.bottom_up_layer(x)
            bu_values.append(x)
        
        input_ = None
        for i in reversed(range(self.nlayers)):
            muq = self.top_down_layer.inference(input_, bu_values[i], i == self.nlayers-1)
            if i == self.nlayers-1:
                mup = th.zeros_like(muq)
            else:
                mup = self.top_down_layer.generative(input_)
            delta_mu = muq - mup
            # quantize
            muz_np = delta_mu.flatten().detach().cpu().numpy()
            cdfss = self.cdfz(muz_np)
            # decompress
            delta_z_ids = coder.decodeSymbols(cdfss)
            delta_z = self.z_center[delta_z_ids]
            delta_z = th.from_numpy(delta_z).to(x.device).float().view(mup.shape)
            z = mup + delta_z
            if zs is None:
                zs = delta_z_ids.copy()
            else:
                zs = np.concatenate((zs, delta_z_ids))
            input_ = self.top_down_layer.deterministic_block(input_, z, i == self.nlayers-1)
        return zs, input_
    
    def _decompress_pz(self, h:int, w:int, coder:MixDecoder, device:str):
        base_size =h*w*self.z_channels
        zsizes = [base_size>>(2*i+2) for i in range(self.nlayers)]
        zshapes = [(1, self.z_channels, h>>(i+1), w>>(i+1)) for i in range(self.nlayers)]
        zsize = np.sum(zsizes)
        z_ids = coder.decodeUniforms(zsize)
        input_ = None
        start_id = 0
        mups = []
        inputs = []
        for i in reversed(range(self.nlayers)):
            inputs.append(input_)
            if i == self.nlayers-1:
                mup = th.zeros(zshapes[i], device=device)
            else:
                mup = self.top_down_layer.generative(input_)
            delta_z_ids = z_ids[start_id:start_id+zsizes[i]]
            start_id += zsizes[i]
            delta_z = self.z_center[delta_z_ids]
            delta_z = th.from_numpy(delta_z).to(device).float().view(mup.shape)
            z = mup + delta_z
            input_ = self.top_down_layer.deterministic_block(input_, z, i == self.nlayers-1)
            mups.append(mup)
            
        return mups[::-1], inputs[::-1], z_ids, input_
    
    def _compress_qz(self, x:th.Tensor, z_ids:np.ndarray, mups:list, inputs:list, coder:MixDecoder):
        _,_,h,w = x.shape
        base_size = h*w*self.z_channels
        zsizes = [base_size>>(2*i+2) for i in range(self.nlayers)]
        zsize = np.sum(zsizes)
        
        x = self.head(x)
        bu_values = []
        for i in range(self.nlayers):
            x = self.bottom_up_layer(x)
            bu_values.append(x)

        end_id = zsize
        for i in range(self.nlayers):
            muq = self.top_down_layer.inference(inputs[i], bu_values[i], i == self.nlayers-1)
            delta_z_ids = z_ids[end_id-zsizes[i]:end_id]
            end_id -= zsizes[i]
            delta_mu = muq - mups[i]
            muz_np = delta_mu.flatten().cpu().__reversed__().detach().numpy()
            delta_z_ids_r = delta_z_ids[::-1].copy()
            cdfss = self.cdfz(muz_np)
            cdfs = cdfss[np.arange(delta_z_ids_r.shape[0]), delta_z_ids_r]
            pmfs = cdfss[np.arange(delta_z_ids_r.shape[0]), delta_z_ids_r + 1] - cdfs
            coder.encodeSymbols(pmfs, cdfs)
        