from config.config import Config
cfg = {
    'share': True,
    'sig_num': 4,
    'n_layers': 3,
    'sig_plane_width': 96,
    'sig_plane_depth': 4,
    'ins_plane_width': 128,
    'ins_plane_depth': 5,
    'latent_width': 128,
    'latent_depth': 8,
    'latent_channels': 3,
    'space_width': 64,
    'entropy_width': 128,
    'entropy_depth': 1,
    'space_width_ins': None,
    'entropy_width_ins': None,
    'entropy_depth_ins': None,
}

CFG = Config(**cfg)