import os
import cv2
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from torch.utils.data import DataLoader
import numpy as np
from metrics import general_dice, general_jaccard,compute_mask_IU

from models.MJXNet import SAM

from tqdm import tqdm
# 新增：导入csv模块，用于保存数据（无需额外安装，Python内置）
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='/opt/data/private/MGX/data',
                    help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='/endovis18',
                    help='Name of Experiment')
parser.add_argument('--num_classes', type=int, default=8,
                    help='output channel of network')
parser.add_argument('--image_size', type=list, default=512,
                    help='patch size of network input')
parser.add_argument('--model',default="save/otherMethods/_cod-sam-vit-b_EMCADNet_18/model_epoch_best.pth")
args = parser.parse_args()


def onehot(data, n):
    flattened_data = data.ravel()
    buf = np.zeros((flattened_data.size, n))
    indices = np.arange(flattened_data.size) * n + flattened_data
    buf.flat[indices] = 1
    return buf.reshape(data.shape + (n,))

def inference(save_path):
    data_transforms = build_transforms(args)

    val_dataset = build_Dataset(args=args, data_dir='endovis18', split="test",
                                transform=data_transforms["valid_test"])
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = SAM().cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=False)
    os.makedirs(save_path, exist_ok=True)

    result_dice = []
    result_jaccard = []
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, args.num_classes)}

    # ===================== 新增：初始化每张图片的challenge IoU记录列表 =====================
    per_image_challenge_iou = []  # 存储格式：[(图片名称, challenge IoU), ...]

    model.eval()
    data_norm=None
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if k != 'name':
                batch[k] = v.cuda()
        image_name = batch['name'][0].split('.')[0]
        test_image = batch['inp']
        test_label = torch.argmax(batch['gt'], dim=1).squeeze(0).detach().cpu().numpy()
        pred = torch.softmax(model.infer(test_image),dim=1)
        pred = torch.argmax(pred, dim=1).squeeze(0).detach().cpu().numpy()
        # 2017计算方式
        result_dice += [general_dice(test_label, pred)]
        result_jaccard += [general_jaccard(test_label, pred)]
        
        im_iou = []
        im_iou_challenge = []
        target = test_label
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] # remove background
        if target.sum() == 0:
            if np.sum(pred) == 0:
                all_im_iou_acc.append(1)
                # 新增：背景图片，challenge IoU记为1（保持逻辑一致）
                img_challenge_iou = 1.0
                per_image_challenge_iou.append((image_name, img_challenge_iou))
            continue
        
        gt_classes = np.unique(test_label)
        # pred = np.where(np.isin(pred, gt_classes), pred, 0)
        for class_id in range(1, args.num_classes): 
            current_pred = (pred == class_id)
            current_target = (test_label == class_id)
            if current_pred.sum() != 0 or current_target.sum() != 0:
                i, u = compute_mask_IU(current_pred, current_target)       
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge+=[(i+1e-15)/(u+1e-15)]
        
        # ===================== 新增：计算当前图片的challenge IoU并记录 =====================
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            img_challenge_iou = np.mean(im_iou_challenge)  # 当前图片的challenge IoU
            all_im_iou_acc_challenge.append(img_challenge_iou)
            per_image_challenge_iou.append((image_name, img_challenge_iou))  # 记录图片名称和对应的IoU
        else:
            # 无有效类别时，challenge IoU记为0（可根据需求调整）
            img_challenge_iou = 0.0
            per_image_challenge_iou.append((image_name, img_challenge_iou))

    final_im_iou = cum_I / cum_U
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)
        
    final_class_im_iou = torch.zeros(9)
    print('Final cIoU per class:')
    print('| Class | cIoU |')
    print('-----------------')
    for c in range(1, args.num_classes):
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).mean()
        print('| {} | {:.5f} |'.format(c, final_class_im_iou[c-1]))
    print('-----------------')
    mean_class_iou = torch.tensor([torch.tensor(values).mean() for c, values in class_ious.items() if len(values) > 0]).mean()
    print('mIoU: {:.5f}, IoU: {:.5f}, challenge IoU: {:.5f}, mean class IoU: {:.5f}'.format(
                                            final_im_iou,
                                            mean_im_iou,
                                            mean_im_iou_challenge,
                                            mean_class_iou))

    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))

    # ===================== 新增：保存每张图片的challenge IoU到CSV文件 =====================
    # CSV文件路径（保存在save_path下，方便管理）
    csv_save_path = os.path.join(save_path, "emacd.csv")
    # 写入CSV文件
    with open(csv_save_path, 'w', newline='', encoding='utf-8') as csvfile:
        # 定义列名
        fieldnames = ['image_name', 'challenge_iou']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        # 写入每张图片的数据
        for img_name, iou_val in per_image_challenge_iou:
            writer.writerow({'image_name': img_name, 'challenge_iou': round(iou_val, 6)})  # 保留6位小数，保证精度

    print(f"\n每张图片的challenge IoU已保存至：{csv_save_path}")


if __name__ == '__main__':
    save_path = None
    save_path = "/opt/data/private/SAM-newAdapter-decoder-otherMethods/testW"
    inference(save_path)