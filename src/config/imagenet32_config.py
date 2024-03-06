from config.config import Config
cfg = {
    'share': False,
    'sig_num': 4,
    'n_layers': 3,
    'sig_plane_width': 192,
    'sig_plane_depth': 6,
    'ins_plane_width': 256,
    'ins_plane_depth': 6,
    'latent_width': 256,
    'latent_depth': 8,
    'latent_channels': 3,
    'space_width': 192,
    'entropy_width': 256,
    'entropy_depth': 3,
    'space_width_ins': None,
    'entropy_width_ins': None,
    'entropy_depth_ins': None,
}

CFG = Config(**cfg)