import os
import sys
import argparse
import torch as th
import numpy as np
import math
from PIL import Image
from utils.transforms import PILToTensorUint8
from torch.utils.data import DataLoader
from modules.arib_bps import ARIB_BPS
from importlib import machinery, util

def parse(args):
    parser = argparse.ArgumentParser(description='Prepare Finetune dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='dir to dataset')
    parser.add_argument('--data_dst', type=str, default=None, help='dir for qp list and imgs, only required for cifar10 dataset')
    parser.add_argument('--dataset_type', type=str, default='filedataset', help='cifar10 | filedataset')
    parser.add_argument('--qp_path', type=str, default='qps.txt', help='path to dataset')
    parser.add_argument('--model', type=str, required=True, help='dir to model')
    parser.add_argument('--config', type=str, required=True, help='path to model config file')
    return parser.parse_args(args)

def main(args):
    args = parse(args)

    # load config
    loader = machinery.SourceFileLoader('config', args.config)
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
    qppath = None
    if args.dataset_type == 'cifar10':
        from torchvision.datasets import CIFAR10
        data_set = CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=PILToTensorUint8(),
        )
        if not os.path.exists(args.data_dst):
            os.makedirs(args.data_dst)
        qppath = os.path.join(args.data_dst, args.qp_path)
    elif args.dataset_type == 'filedataset':
        from utils.datasets import FileDataset
        data_set = FileDataset(
            dir=args.data_dir,
            transform=PILToTensorUint8(),
        )
        qppath = os.path.join(args.data_dir, args.qp_path)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

    print(f'find {len(data_set)} images')

    qps = np.zeros(len(data_set)).astype(int)
    with th.no_grad():
        for idx1, (x, idx2) in enumerate(data_loader):
            x = x.to(device)
            xsize = np.prod(x.shape[1:])
            zsize = 0.25*((1-0.25**model.n_layers)/0.75)*x.shape[2]*x.shape[3]*model.z_channels
            if args.dataset_type == 'cifar10':
                loss_ins = model.ins.inference(x).sum()
                qp_ = loss_ins.item()*xsize/zsize
                qp = min(max(model.ql, qp_), model.qu)
                qps[idx1] = int(math.floor(qp))
                img = x[0,...].cpu().detach().numpy().transpose(1,2,0)
                img = Image.fromarray(img.astype(np.uint8))
                img.save(os.path.join(args.data_dst, f'{idx1}.png'))
            else:
                qp_ = loss_ins.item()*xsize/zsize
                qp_ = loss_ins*xsize/zsize
                qp = np.floor(qp_).astype(int).clip(model.ql, model.qu)
                qps[idx2] = np.floor(qp).astype(int)
    print(f'saving qplist to {qppath}')
    np.savetxt(qppath, qps)

if __name__ == '__main__':
    main(sys.argv[1:])