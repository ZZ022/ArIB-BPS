# from torchvision.utils import _log_api_usage_once
import numpy as np
from utils.helpers import get_bit_plane
import torch
from torchvision.utils import _log_api_usage_once

# def _log_api_usage_once(x):
#     return

class PILToTensorUint8:
    def __init__(self) -> None:
        _log_api_usage_once(self)

    # @staticmethod
    def __call__(self, pic):     
        return torch.Tensor(np.array(pic)).to(torch.uint8).permute((2,0,1)).contiguous()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ToTensorUint8:
    def __init__(self) -> None:
        _log_api_usage_once(self)

    # @staticmethod
    def __call__(self, pic):     
        return torch.Tensor(pic).to(torch.uint8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class GetSubPlane:
    def __init__(self, plane:int) -> None:
        _log_api_usage_once(self)
        self.plane = plane

    # @staticmethod
    def __call__(self, x):     
        return get_bit_plane(x, self.plane)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
