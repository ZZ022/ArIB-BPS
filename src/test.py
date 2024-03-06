import os
import sys
import argparse
import torch as th
from utils.transforms import PILToTensorUint8
from torch.utils.data import DataLoader
from modules.arib_bps import ARIB_BPS
import math

def parse(args):
    parser = argparse.ArgumentParser(description='ARIB-BPS test')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 | imagenet32 | imagenet64 | imagenet64_small')
    parser.add_argument('--dataset_type', type=str, default='imagefolder', help='cifar10 | imagefolder| filedataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='dir to dataset')
    parser.add_argument('--model', type=str, default='./model', help='dir to model')
    parser.add_argument('--log_path', type=str, default='./log.txt', help='path to log')
    parser.add_argument('--mode', type=str, required=True, help='inference | single | dataset | speed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size or size of dataset for dataset compression setting')
    return parser.parse_args(args)

def main(args):
    args = parse(args)
    if args.dataset == 'cifar10':
        from config.cifar_config import CFG
    elif args.dataset == 'imagenet32':
        from config.imagenet32_config import CFG
    elif args.dataset == 'imagenet64':
        from config.imagenet64_config import CFG
    elif args.dataset == 'imagenet64_small':
        from config.imagenet64_small_config import CFG
    else:
        raise ValueError('Invalid dataset name')
    
    if args.dataset_type == 'cifar10':
            from torchvision.datasets import CIFAR10
            test_set = CIFAR10(
                root=args.data_dir,
                train=False,
                download=True,
                transform=PILToTensorUint8(),
            )
    elif args.dataset_type == 'imagefolder':
        from torchvision.datasets import ImageFolder
        test_set = ImageFolder(
            root=args.data_dir,
            transform=PILToTensorUint8(),
        )
    elif args.dataset_type == 'filedataset':
        from utils.datasets import FileDataset
        test_set = FileDataset(
            dir=args.data_dir,
            transform=PILToTensorUint8(),
        )

    batchsize = args.batch_size if args.mode == 'inference' or args.mode == 'speed' else 1
    test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)

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
    if args.mode == 'inference':
        print('inference mode')
        test_inference(model, test_loader, int(math.ceil(len(test_set)/args.batch_size)), device)
    elif args.mode == 'single':
        print('single-image compression mode')
        print('log path:', args.log_path)
        test_single(model, test_loader, args.log_path, device)
    elif args.mode == 'dataset':
        print('dataset compression mode')
        print('log path:', args.log_path)
        test_dataset(model, test_loader, args.batch_size, args.log_path, device)
    elif args.mode == 'speed':
        print('speed test mode')
        test_speed(model, test_loader, device)
    else:
        raise ValueError('Invalid mode name')

def test_inference(model, test_loader, total_step, device):
    from tqdm import tqdm
    losses_sum = 0
    num_data = 0
    with tqdm(total=total_step, desc='inference') as pbar:
        with th.no_grad():
            for _, (x, _) in enumerate(test_loader):
                x = x.to(device)
                b = x.shape[0]
                losses = model.inference(x).sum().item()
                losses_sum += losses * b
                num_data += b
                log = f'{losses_sum/num_data:.4f}'
                pbar.set_postfix_str(log)
                pbar.update(1)
    print(f'loss: {losses_sum/num_data:.4f}')

def test_single(model, test_loader, log_path, device):
    import pandas as pd
    import numpy as np
    import random
    th.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    with th.no_grad():
        with open(log_path, 'a') as f:
            f.write('img_size,stream_size,enc_time,dec_time,k,aux_bpd,errsum\n')
            for _, (x, _) in enumerate(test_loader):
                x = x.to(device)
                f.write(model.eval_single(x, 16) +'\n')
                f.flush()
    
    data_frame = pd.read_csv(log_path)
    bpd = float(data_frame['stream_size'].sum())/data_frame['img_size'].sum()
    num_decode_failure = sum(data_frame['errsum'] != 0)
    print(f'avg_bpd: {bpd:.4f} num_decode_failure: {num_decode_failure}')

def test_dataset(model, test_loader, batch_size, log_path, device):
    import pandas as pd
    import numpy as np
    import random
    th.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    xs = []
    with th.no_grad():
        with open(log_path, 'a') as f:
            f.write('dataset_size,stream_size,enc_time,dec_time,errsum\n')
            for i, (x, _) in enumerate(test_loader):
                x = x.to(device)
                xs.append(x)
                if i%batch_size == batch_size-1:
                    f.write(model.eval_dataset(xs, 16, 10)+'\n')
                    xs = []
                    f.flush()
            if len(xs) != 0:
                f.write(model.eval_dataset(xs, 16, 10)+'\n')
                f.flush()

    data_frame = pd.read_csv(log_path)
    bpd = float(data_frame['stream_size'].sum())/data_frame['dataset_size'].sum()
    num_decode_failure = sum(data_frame['errsum'] != 0)
    print(f'avg_bpd: {bpd:.4f} num_decode_failure: {num_decode_failure}')

def test_speed(model, test_loader, device):
    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    avg_inf_time_sum = 0
    num_iter = 3
    num_test = 100
    b = 0
    for j in range(num_iter):
        inf_time_sum = 0
        with th.no_grad():
            for i, (x, _) in enumerate(test_loader):
                x = x.to(device)
                # warm up
                if i == 0 and j == 0:
                    [model.inference(x) for _ in range(10)]
                    b = x.shape[0]
                if i == num_test:
                    break
                start.record()
                model.inference(x)
                end.record()
                th.cuda.synchronize()
                inf_time_sum += start.elapsed_time(end)
        avg_inf_time_sum += inf_time_sum/num_test/b
    print(f'inf_time: {avg_inf_time_sum/num_iter:.2f} ms/sample')

if __name__ == '__main__':
    main(sys.argv[1:])
