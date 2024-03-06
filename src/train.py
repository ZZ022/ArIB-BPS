import os
import sys
import logging
import datetime
import shutil
import argparse

import torch as th
import numpy as np

from importlib import machinery, util
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader, Subset
from utils.transforms import PILToTensorUint8
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import torch.multiprocessing as mp
from modules.arib_bps import SIG, INS
from utils.datasets import FileDataset, FinetuneDataset

def get_time():
    return str(datetime.datetime.now())[0:19]

def set_logger(log):
    level = getattr(logging, 'INFO', None)
    handler = logging.FileHandler(log)
    formatter = logging.Formatter('')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)

def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def parse(args):
    parser = argparse.ArgumentParser(description='ARIB-BPS train')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--mode', type=str, required=True, help='sig | ins | finetune')
    parser.add_argument('--dataset_type', type=str, default='imagefolder', help='cifar10 | imagefolder | filedataset')
    parser.add_argument('--data_dir', type=str, required=True, help='dir to dataset')
    parser.add_argument('--qp_path', type=str, default=None, help='path to qp list')
    parser.add_argument('--valid_num', type=int, default=5000, help='size of validation set')
    parser.add_argument('--resume', type=str, default='', help='path to pretrained model')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save')
    parser.add_argument('--config', type=str, required=True, help='path to model config')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_iters', type=int, default=1e6, help='number of iterations')
    parser.add_argument('--log_interval', type=int, default=100, help='interval of logging')
    parser.add_argument('--valid_interval', type=int, default=1000, help='interval of valid')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout p')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='decay rate')
    parser.add_argument('--decay_interval', type=int, default=1000, help='interval of decay')
    parser.add_argument('--master_port', type=str, default='12345', help='master port for ddp')
    return parser.parse_args(args)

def main(args):
    args = parse(args)
    
    # load config
    loader = machinery.SourceFileLoader('config', args.config)
    spec = util.spec_from_loader(loader.name, loader)
    config = util.module_from_spec(spec)
    loader.exec_module(config)
    CFG = config.CFG
    
    prefix = f'{get_time()}_{args.mode}'

    if not os.path.exists(os.path.join(args.save_dir, prefix)):
         os.mkdir(os.path.join(args.save_dir, prefix))
    log_dir = os.path.join(args.save_dir, prefix, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    ckpt_dir = os.path.join(args.save_dir, prefix, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    # copy config to save dir
    shutil.copy(args.config, os.path.join(args.save_dir, prefix, 'config.py'))

    if args.mode == 'sig':
        train_sig(args, CFG, log_dir, ckpt_dir)
    elif args.mode == 'ins':
        train_ins(args, CFG, log_dir, ckpt_dir)
    elif args.mode == 'finetune':
        train_finetune(args, CFG, log_dir, ckpt_dir)
    else:
        raise ValueError('Invalid mode')

def train_sig(args, cfg, log_dir, ckpt_dir):
    world_size = args.num_gpus
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(_train_sig_ddp_single_task, args=(world_size, args, cfg, log_dir, ckpt_dir, False), nprocs=world_size)

def train_ins(args, cfg, log_dir, ckpt_dir):
    world_size = args.num_gpus
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(_train_ins_ddp_single_task, args=(world_size, args, cfg, log_dir, ckpt_dir), nprocs=world_size)

def train_finetune(args, cfg, log_dir, ckpt_dir):
    world_size = args.num_gpus
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(_train_sig_ddp_single_task, args=(world_size, args, cfg, log_dir, ckpt_dir, True,), nprocs=world_size)

def _train_sig_ddp_single_task(rank, world_size, args, cfg, log_dir, ckpt_dir, finetune):
    os.environ['RANK'] = str(rank)
    th.cuda.set_device(rank)
    device = th.device('cuda', rank)
    dist.init_process_group(backend='nccl', world_size=world_size, init_method='env://')
    is0 = rank == 0
    bpd_min = cfg.sig_num

    if finetune:
        all_set = FinetuneDataset(args.data_dir, qp_path=args.qp_path, transform=PILToTensorUint8())
    else:
        if args.dataset_type == 'cifar10':
            all_set = CIFAR10(args.data_dir, train=True, transform=PILToTensorUint8(), download=True)
        elif args.dataset_type == 'imagefolder':
            all_set = ImageFolder(args.data_dir, transform=PILToTensorUint8())
        elif args.dataset_type == 'filedataset':
            all_set = FileDataset(args.data_dir, transform=PILToTensorUint8())
    length = len(all_set)
    train_set = Subset(all_set, range(0, length-args.valid_num))
    valid_set = Subset(all_set, range(length-args.valid_num, length))
    train_sampler = DistributedSampler(train_set)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=True, shuffle=False)

    model = SIG(
        sig_num=cfg.sig_num,
        n_layers=cfg.n_layers,
        plane_width=cfg.sig_plane_width,
        plane_depth=cfg.sig_plane_depth,
        latent_width=cfg.latent_width,
        latent_depth=cfg.latent_depth,
        latent_channels=cfg.latent_channels,
        space_width=cfg.space_width,
        entropy_width=cfg.entropy_width,
        entropy_depth=cfg.entropy_depth,
        share=cfg.share,
        dropout=args.dropout,
    )

    if os.path.exists(args.resume):
        model.load_state_dict(th.load(args.resume, map_location='cpu'))
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay_rate)
    if is0:
        set_logger(os.path.join(log_dir, 'train.log'))
        logging.info(args)
        logging.info(f'use {world_size} gpus')
    model.train()
    num_iters = 0
    losses = []
    epoch_id = 0

    while num_iters < args.num_iters:
        train_sampler.set_epoch(epoch_id)
        for x, qps in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            if not finetune:
                loss = model(x).sum()
            else:
                loss = model(x, qps).sum()
            loss.backward()
            optimizer.step()
            num_iters += 1
            losses.append(loss.item())
            if num_iters % args.log_interval == 0 and is0:
                logging.info(f'{get_time()} | train | {num_iters} | {np.mean(losses):.4f}')
                losses = []
            if num_iters % args.valid_interval == 0:
                model.eval()
                with th.no_grad():
                    bpd = []
                    valid_num = 0
                    for x, qps in valid_loader:
                        x = x.to(device)
                        if not finetune:
                            bpds = model(x)
                        else:
                            bpds = model(x, qps)
                        bpds = reduce_mean(bpds, world_size)
                        bpd.append(bpds.sum().item())
                        valid_num += 1
                    bpd = np.mean(bpd)
                    if is0:
                        logging.info(f'{get_time()} | valid | {num_iters} | {bpd:.4f}')
                    if bpd < bpd_min and is0:
                        bpd_min = bpd
                        th.save(model.module.state_dict(), os.path.join(ckpt_dir, 'sig.pth'))
                if is0:
                    th.save(model.module.state_dict(), os.path.join(ckpt_dir, f'sig_latest.pth'))
                model.train()
            if num_iters % args.decay_interval == 0:
                lr_scheduler.step()
                if is0:
                    logging.info(f'lr decay to {lr_scheduler.get_lr()[0]}')
            if num_iters >= args.num_iters:
                break
        epoch_id += 1

def _train_ins_ddp_single_task(rank, world_size, args, cfg, log_dir, ckpt_dir):
    os.environ['RANK'] = str(rank)
    th.cuda.set_device(rank)
    device = th.device('cuda', rank)
    dist.init_process_group(backend='nccl', world_size=world_size, init_method='env://')
    is0 = rank == 0
    bpd_min = 8 - cfg.sig_num

    if args.dataset_type == 'cifar10':
        all_set = CIFAR10(args.data_dir, train=True, transform=PILToTensorUint8(), download=True)
    elif args.dataset_type == 'imagefolder':
        all_set = ImageFolder(args.data_dir, transform=PILToTensorUint8())
    elif args.dataset_type == 'filedataset':
        all_set = FileDataset(args.data_dir, transform=PILToTensorUint8())
    length = len(all_set)
    train_set = Subset(all_set, range(0, length-args.valid_num))
    valid_set = Subset(all_set, range(length-args.valid_num, length))
    train_sampler = DistributedSampler(train_set)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=True, shuffle=False)

    if cfg.space_width_ins is None:
        model = INS(
            sig_num=cfg.sig_num,
            n_layers=cfg.n_layers + 1,
            plane_width=cfg.ins_plane_width,
            plane_depth=cfg.ins_plane_depth,
            space_width=cfg.space_width,
            entropy_width=cfg.entropy_width,
            entropy_depth=cfg.entropy_depth,
            share=cfg.share,
            dropout=args.dropout,
        )
    else:
        model = INS(
            sig_num=cfg.sig_num,
            n_layers=cfg.n_layers + 1,
            sig_plane_width=cfg.sig_plane_width,
            sig_plane_depth=cfg.sig_plane_depth,
            space_width=cfg.space_width_ins,
            entropy_width=cfg.entropy_width_ins,
            entropy_depth=cfg.entropy_depth_ins,
            dropout=args.dropout,
        )

    if os.path.exists(args.resume):
        model.load_state_dict(th.load(args.resume, map_location='cpu'))
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay_rate)
    if is0:
        set_logger(os.path.join(log_dir, 'train.log'))
        logging.info(args)
        logging.info(f'use {world_size} gpus')
    model.train()
    num_iters = 0
    losses = []
    epoch_id = 0

    while num_iters < args.num_iters:
        train_sampler.set_epoch(epoch_id)
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            num_iters += 1
            losses.append(loss.item())
            if num_iters % args.log_interval == 0 and is0:
                logging.info(f'{get_time()} | train | {num_iters} | {np.mean(losses):.4f}')
                losses = []
            if num_iters % args.valid_interval == 0:
                model.eval()
                with th.no_grad():
                    bpd = []
                    valid_num = 0
                    for x, _ in valid_loader:
                        x = x.to(device)
                        bpds = model(x)
                        bpds = reduce_mean(bpds, world_size)
                        bpd.append(bpds.sum().item())
                        valid_num += 1
                    bpd = np.mean(bpd)
                    if is0:
                        logging.info(f'{get_time()} | valid | {num_iters} | {bpd:.4f}')
                    if bpd < bpd_min and is0:
                        bpd_min = bpd
                        th.save(model.module.state_dict(), os.path.join(ckpt_dir, 'ins.pth'))
                if is0:
                    th.save(model.module.state_dict(), os.path.join(ckpt_dir, f'ins_latest.pth'))
                model.train()
            if num_iters % args.decay_interval == 0:
                lr_scheduler.step()
                if is0:
                    logging.info(f'lr decay to {lr_scheduler.get_lr()[0]}')
            if num_iters >= args.num_iters:
                break
        epoch_id += 1

if __name__ == '__main__':
    main(sys.argv[1:])