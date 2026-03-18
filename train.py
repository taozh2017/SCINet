import argparse
import os
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
import utils
from statistics import mean
import torch
import torch.distributed as dist
import random
from models.doTrain import Trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:25678', world_size=1, rank=0)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def eval_psnr(loader, model):
    model.eval()
    metric_fn = utils.calc_muti
    metric1, metric2 = 'Dice', 'Jaccard'
    pbar = tqdm(loader, leave=False, desc='val')

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    for batch in pbar:
        for k, v in batch.items():
            if k != 'name':
                batch[k] = v.cuda()
        inp = batch['inp']
        pred = model.infer(inp)
        result1, result2 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
    return val_metric1.item(), val_metric2.item(),metric1, metric2,



def prepare_training(max_epoch):
    model = Trainer().cuda()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    epoch_start = 1
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=1.0e-7)
    if local_rank == 0:
        log('model: #total_params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model,args):
    model.train()
    pbar = tqdm(total=len(train_loader), leave=False, desc='train')   
    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            if k != 'name':
                batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']    
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(save_path, args):
    global log, writer, log_info
    log, writer = utils.set_save_path(save_path, remove=False)

    data_transforms = build_transforms(args)
    train_dataset = build_Dataset(args=args, data_dir=args.dataset, split="train",
                                  transform=data_transforms["train"])
    val_dataset = build_Dataset(args=args, data_dir=args.dataset, split="val",
                                transform=data_transforms["valid_test"])

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=2, pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)


    model, optimizer, epoch_start, lr_scheduler = prepare_training(args.epoch_max)
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, args.epoch_max, eta_min=1.0e-7)

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = args.epoch_max
    epoch_val = 1
    max_val_v = -1e18 # 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model,args)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f},lr={:.4f}'.format(train_loss_G,optimizer.param_groups[0]['lr']))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            save( model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, metric1, metric2 = eval_psnr(val_loader, model)

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)

                if result2 > max_val_v:
                    max_val_v = result2
                    save(model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()


def save(model, save_path, name):
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='8')
    parser.add_argument('--model',default="/model_epoch_best.pth")
    parser.add_argument('--tag', default='BAANet_17_new')
    parser.add_argument('--dataset', default='endovis17')
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--num_classes", type=int, default=8, help="")
    parser.add_argument("--epoch_max", type=int, default=200, help="")
    parser.add_argument("--image_size", type=int, default=512, help="")
    args = parser.parse_args()

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('/save/otherMethods', save_name)

    main(save_path, args=args)
