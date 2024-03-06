from dataclasses import dataclass

@dataclass
class Config:
    share: bool
    sig_num: int
    n_layers: int
    sig_plane_width: int
    sig_plane_depth: int
    ins_plane_width: int
    ins_plane_depth: int
    latent_width: int
    latent_depth: int
    latent_channels: int
    space_width: int
    entropy_width: int
    entropy_depth: int
    space_width_ins: int
    entropy_width_ins: int
    entropy_depth_ins: int