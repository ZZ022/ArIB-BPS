import torch as th
import torch.nn as nn
import numpy as np
from modules.wnres import softplus
from utils.coder.mixcoder import MixEncoder, MixDecoder

bin_precision = 8
bin_scale = float(1 << bin_precision)
lb = 1e-6
half_scale = float(1 << (bin_precision - 1))

class BinQuant(th.autograd.Function):
    def forward(ctx, input):
        return th.round(input*bin_scale).clamp(1.0,bin_scale-1.0)/bin_scale

    def backward(ctx, grad_output):
        return grad_output
    
def binquant(x:th.Tensor, train:bool=True) -> th.Tensor:
    if train:
        return x.clamp(lb, 1 - lb)
    else:
        return BinQuant.apply(x)

def get_bit_plane(x:th.Tensor, idx) -> th.Tensor:
    return ((x >> (8-idx)) & 1).float()

# function to sample from a Logistic distribution (mu=0, scale=1), from BitSwap
def logistic_eps(shape:th.Tensor, device, bound=1e-5)->th.Tensor:
    # sample from a Gaussian
    u = th.rand(shape, device=device)

    # clamp between two bounds to ensure numerical stability
    u = th.clamp(u, min=bound, max=1 - bound)

    ## sigmoid^(-1)(u) = log(u)-log(1-u)
    # transform to a sample from the Logistic distribution
    eps = th.log(u) - th.log1p(-u)
    return eps

# modified from BitSwap
def logistic_logp(mu:th.Tensor, x:th.Tensor)->th.Tensor:
    _y = -(x - mu)
    _logp = -_y - 2 * softplus(-_y)
    return _logp

def rgb_loss(params:th.Tensor, target:th.Tensor)->th.Tensor:
    pr, pg, pb, alpha, beta, gamma = th.chunk(params, 6, dim=1)
    r, g, _ = th.chunk(2*target-1, 3, dim=1)
    pr = binquant(th.sigmoid(pr))
    pg = binquant(th.sigmoid(pg+r*alpha))
    pb = binquant(th.sigmoid(pb+r*beta+g*gamma))
    p1 = th.cat([pr, pg, pb], dim=1)
    return -th.sum(th.log2(p1)*target+th.log2(1-p1)*(1-target), dim=(0,2,3))

def get_p1(params:th.Tensor, target:th.Tensor)->th.Tensor:
    pr, pg, pb, alpha, beta, gamma = th.chunk(params, 6, dim=1)
    r, g, _ = th.chunk(2*target-1, 3, dim=1)
    pr = th.sigmoid(pr)
    pg = th.sigmoid(pg+r*alpha)
    pb = th.sigmoid(pb+r*beta+g*gamma)
    p1 = th.cat([pr, pg, pb], dim=1)
    return p1*bin_scale

def get_i1(params:th.Tensor, target:th.Tensor)->th.Tensor:
    return th.round(get_p1(params, target)).clamp(1, bin_scale - 1).to(th.uint8)

def encode_with_p1(p1:th.Tensor, bins:th.Tensor, coder:MixEncoder)->None:
    p1 = p1.flatten().__reversed__()
    is0Lsb = p1>=half_scale
    lsp = th.where(p1<=half_scale, p1, bin_scale-p1)
    bins = bins.flatten().__reversed__().to(th.uint8)
    isLsb = th.logical_xor(is0Lsb, bins)
    lsp_np = lsp.detach().cpu().numpy().copy().astype(np.uint8) - 1
    isLsb_np = isLsb.detach().cpu().numpy().copy().astype(np.uint8)
    coder.encodeBins(isLsb_np, lsp_np)

def decode_with_p1(p1:th.Tensor, coder:MixDecoder)->np.ndarray:
    is0Lsb = p1>=half_scale
    lsp = th.where(p1<=half_scale, p1, bin_scale-p1)
    lsp_np = lsp.flatten().detach().cpu().numpy().copy().astype(np.uint8) - 1
    isLsp = coder.decodeBins(lsp_np)
    isLsp = th.from_numpy(isLsp).reshape(p1.shape).to(p1.device).to(th.uint8)
    return th.logical_xor(is0Lsb, isLsp).float()

def decode_with_params(params:th.Tensor, coder:MixDecoder):
    pr, pg, pb, alpha, beta, gamma = th.chunk(params, 6, dim=1)
    f = lambda x: th.round(th.sigmoid(x)*bin_scale).clamp(1,bin_scale-1).to(th.uint8)
    ir = f(pr)
    r = decode_with_p1(ir, coder)
    ig = f(pg+(2*r-1)*alpha)
    g = decode_with_p1(ig, coder)
    ib = f(pb+(2*r-1)*beta+(2*g-1)*gamma)
    b = decode_with_p1(ib, coder)
    return th.cat([r,g,b], dim=1)

def fake_discretize(x:th.Tensor, qp:int):
    scale = float(1<<qp)
    ste_element = scale*th.sigmoid(x)
    temp = th.ceil(ste_element).detach() + ste_element - ste_element.detach() - 0.5
    return -th.log((scale/temp-1.0).clamp(1e-6))

def fake_sample(mup:th.Tensor, muq:th.Tensor, qp:int):
    scale = float(1<<qp)
    eps = logistic_eps(muq.shape, muq.device)
    delta = eps + muq - mup
    ste_element = scale*th.sigmoid(delta)

    _cdf_low = th.ceil(ste_element).detach() + ste_element - ste_element.detach() - 1.0
    cdf_low = _cdf_low + lb * scale

    cdf_mid = _cdf_low + 0.5 
    z_hat = -th.log(scale/cdf_mid-1.0) + mup

    z_low = -th.log(scale/cdf_low-1.0) + mup - muq
    cdf_up = _cdf_low + 1.0 - lb * scale
    z_up = -th.log(scale/cdf_up-1.0) + mup - muq

    cdf_l = th.sigmoid(z_low)
    cdf_u = th.sigmoid(z_up)

    log_qz = th.sum(th.log2(cdf_u - cdf_l + lb))
    log_pz = qp*np.prod(z_hat.shape)
    return z_hat, log_pz, log_qz