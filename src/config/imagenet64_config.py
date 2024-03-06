from config.config import Config
cfg = {
    'share': False,
    'sig_num': 4,
    'n_layers': 3,
    'sig_plane_width': 160,
    'sig_plane_depth': 3,
    'ins_plane_width': 192,
    'ins_plane_depth': 5,
    'latent_width': 192,
    'latent_depth': 8,
    'latent_channels': 3,
    'space_width': 128,
    'entropy_width': 192,
    'entropy_depth': 2,
    'space_width_ins': 160,
    'entropy_width_ins': 192,
    'entropy_depth_ins': 2,
}

CFG = Config(**cfg)