# adapted from bitswap: https://github.com/fhkingma/bitswap/blob/master/utils/torch/rand.py

import torch
import numpy as np
import modules.wnres as modules


# function to transform "noise" using a given mean and scale
def transform(eps, mu, scale):
    sample = mu + scale * eps
    return sample

# function to sample from a Logistic distribution (mu=0, scale=1)
def logistic_eps(shape, device, bound=1e-5):
    # sample from a Gaussian
    u = torch.rand(shape, device=device)

    # clamp between two bounds to ensure numerical stability
    u = torch.clamp(u, min=bound, max=1 - bound)

    ## sigmoid^(-1)(u) = log(u)-log(1-u)
    # transform to a sample from the Logistic distribution
    eps = torch.log(u) - torch.log1p(-u)
    return eps

# function to calculate the log-probability of x under a Logistic(mu, scale) distribution
def logistic_logp(mu, scale, x):
    _y = -(x - mu) / scale
    _logp = -_y - torch.log(scale) - 2 * modules.softplus(-_y)
    logp = _logp.flatten(2)
    return logp

#function to calculate the log-probability of x under a discretized Logistic(mu, scale) distribution
#heavily based on discretized_mix_logistic_loss() in https://github.com/openai/pixel-cnn
def discretized_logistic_logp(mu, scale, x, bitdepth=8):
    # [0,255] -> [-1.1] (this means bin sizes of 2./255.)
    max_v = float((1<<bitdepth)-1)
    mid = max_v/2.0
    x_rescaled = (x - mid) / mid
    invscale = 1. / scale

    x_centered = x_rescaled - mu

    plus_in = invscale * (x_centered + 1. / max_v)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = invscale * (x_centered - 1. / max_v)
    cdf_min = torch.sigmoid(min_in)

    # log-probability for edge case of 0
    log_cdf_plus = plus_in - modules.softplus(plus_in)

    # log-probability for edge case of 255
    log_one_minus_cdf_min = - modules.softplus(min_in)

    # other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = invscale * x_centered

    # log-probability in the center of the bin, to be used in extreme cases
    log_pdf_mid = mid_in - torch.log(scale) - 2. * modules.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal case, extremely low-probability case
    cond1 = torch.where(cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-12, max=None)),
                        log_pdf_mid - np.log(mid))
    cond2 = torch.where(x_rescaled > .999, log_one_minus_cdf_min, cond1)
    logps = torch.where(x_rescaled < -.999, log_cdf_plus, cond2)

    logp = logps
    return logp

# function to calculate the CDF of the Logistic(mu, scale) distribution evaluated under x
def logistic_cdf(x, mu, scale):
    return torch.sigmoid((x - mu) / scale)

# function to calculate the inverse CDF (quantile function) of the Logistic(mu, scale) distribution evaluated under x
def logistic_icdf(p, mu, scale):
    return mu + scale * torch.log(p / (1. - p))

# class that is used to determine endpoints and centers of discretization bins
# in which every bin has equal mass under some given Logistic(mu, scale) distribution.
# note: the first (-inf) and last (inf) endpoint are not created here, but rather
# accounted for in the compression/decompression loop
class Bins:
    def __init__(self, mu, scale, precision):
        # number of bits used
        self.precision = precision

        # the resulting number of bins from the amount of bits used
        self.nbins = 1 << precision

        # parameters of the Logistic distribution
        self.mu, self.scale = mu, scale

        # datatype used
        self.type = self.mu.dtype

        # device used (GPU/CPU)
        self.device = self.mu.device
        self.shape = list(self.mu.shape)

    def endpoints(self):
        # first uniformly between [0,1]
        # shape: [1 << bits]
        endpoint_probs = torch.arange(1., self.nbins, dtype=self.type, device=self.device) / self.nbins

        # reshape
        endpoint_probs = endpoint_probs[(None,) * len(self.shape)] # shape: [1, 1, 1<<bits]
        endpoint_probs = endpoint_probs.permute([-1] + list(range(len(self.shape)))) # shape: [1 << bits, 1, 1]
        endpoint_probs = endpoint_probs.expand([-1] + self.shape) # shape: [1 << bits] + self.shape

        # put those samples through the inverse CDF
        endpoints = logistic_icdf(endpoint_probs, self.mu, self.scale)

        # reshape
        endpoints = endpoints.permute(list(range(1, len(self.shape) + 1)) + [0]) # self.shape + [1 << bits]
        return endpoints

    def centres(self):
        # first uniformly between [0,1]
        # shape: [1 << bits]
        centre_probs = (torch.arange(end=self.nbins, dtype=self.type, device=self.device) + .5) / self.nbins

        # reshape
        centre_probs = centre_probs[(None,) * len(self.shape)] # shape: [1, 1, 1<<bits]
        centre_probs = centre_probs.permute([-1] + list(range(len(self.shape)))) # shape: [1 << bits, 1, 1]
        centre_probs = centre_probs.expand([-1] + self.shape) # shape: [1 << bits] + self.shape

        # put those samples through the inverse CDF
        centres = logistic_icdf(centre_probs, self.mu, self.scale)

        # reshape
        centres = centres.permute(list(range(1, len(self.shape) + 1)) + [0]) # self.shape + [1 << bits]
        return centres
