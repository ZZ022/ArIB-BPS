import os
import sys
import argparse
from importlib import machinery, util
import torch as th
import numpy as np
from PIL import Image
from modules.arib_bps import ARIB_BPS


def parse(args):
    parser = argparse.ArgumentParser(description='ARIB-BPS compress')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--encode', '-e', action='store_true', help='Encode mode')
    group.add_argument('--decode', '-d', action='store_true', help='Decode mode')
    parser.add_argument('--input','-i', type=str, required=True, help='input file')
    parser.add_argument('--output','-o', type=str, required=True, help='output file')
    parser.add_argument('--config','-c', type=str, required=True, help='model config')
    parser.add_argument('--model','-m', type=str, required=True, help='model dir')
    return parser.parse_args(args)

def main(args):
    args = parse(args)
    config_path = args.config
    
    loader = machinery.SourceFileLoader('config', config_path)
    spec = util.spec_from_loader(loader.name, loader)
    config = util.module_from_spec(spec)
    loader.exec_module(config)
    CFG = config.CFG

    model = ARIB_BPS(
        sig_num=CFG.sig_num,
        n_layers=CFG.n_layers,
        sig_plane_width=CFG.sig_plane_width,
        sig_plane_depth=CFG.sig_plane_depth,
        ins_plane_width=CFG.ins_plane_width,
        ins_plane_depth=CFG.ins_plane_depth,
        latent_width=CFG.latent_width,
        latent_depth=CFG.latent_depth,
        latent_channels=CFG.latent_channels,
        space_width=CFG.space_width,
        entropy_width=CFG.entropy_width,
        entropy_depth=CFG.entropy_depth,
        space_width_ins=CFG.space_width_ins,
        entropy_width_ins=CFG.entropy_width_ins,
        entropy_depth_ins=CFG.entropy_depth_ins,
        share=CFG.share,
    )

    model.load(os.path.join(args.model, 'sig.pth'), os.path.join(args.model, 'ins.pth'))
    model.eval()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model.to(device)

    if args.encode:
        print('Encoding')
        img = Image.open(args.input)
        img_array = np.array(img).transpose(2, 0, 1)
        img_tensor = th.tensor(img_array).to(th.uint8).unsqueeze(0).to(device)
        with th.no_grad():
            model.compress_to_file(img_tensor, args.output)
    elif args.decode:
        print('Decoding')
        with th.no_grad():
            model.decompress_from_file(args.input, args.output, device)

if __name__ == '__main__':
    main(sys.argv[1:])