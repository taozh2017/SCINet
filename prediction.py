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

from models.DoNet import DoNet
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='/opt/data/private/MGX/data',
                    help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='/endovis18',
                    help='Name of Experiment')
parser.add_argument('--num_classes', type=int, default=8,
                    help='output channel of network')
parser.add_argument('--in_channels', type=int, default=3,
                    help='input channel of network')
parser.add_argument('--image_size', type=list, default=512,
                    help='patch size of network input')
parser.add_argument('--model',default="/opt/data/private/save/lightmodel/mynetSCTNet.pth")
parser.add_argument('--save_preds',default="/opt/data/private/save/lightmodel/preds")
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()


def onehot(data, n):
    flattened_data = data.ravel()
    buf = np.zeros((flattened_data.size, n))
    indices = np.arange(flattened_data.size) * n + flattened_data
    buf.flat[indices] = 1
    return buf.reshape(data.shape + (n,))

def inference():
    data_transforms = build_transforms(args)

    val_dataset = build_Dataset(args=args, data_dir='endovis18', split="test",
                                transform=data_transforms["valid_test"])
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = DoNet().cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=False)
    

    result_dice = []
    result_jaccard = []
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, args.num_classes)}

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

        # if args.save_preds:
        #     save_vis(test_image, pred, test_label, args.save_preds, image_name)
        im_iou = []
        im_iou_challenge = []
        target = test_label
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] # remove background
        if target.sum() == 0:
            if np.sum(pred) == 0:
                all_im_iou_acc.append(1)
            continue
        
        gt_classes = np.unique(test_label)
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
                    # consider only classes present in gt
                    im_iou_challenge+=[(i+1e-15)/(u+1e-15)]
        if len(im_iou) > 0:
            # to avoid nans by appending empty list
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            # to avoid nans by appending empty list
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

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


# color_list = [
# [0.1, 0.8, 0.1],
# [0.2, 0.7, 0.2],
# [0.3, 0.6, 0.3],
# [0.4, 0.5, 0.4],
# [0.5, 0.4, 0.5],
# [0.6, 0.3, 0.6],
# [0.7, 0.2, 0.7],
# [0.8, 0.1, 0.8],
#               ]
color_list = [
    [0.0, 0.0, 0.0],   # 色
    [255,0,0],   # 红色
    [255 ,127 ,0],   # 橙色
    [255, 255, 0],   # 黄色
    [0, 255, 0],   # 绿色
    [105, 105, 105],   # hui色
    [0 ,0 ,255],   # 蓝色
    [255 ,0, 255]    # 紫色
]
color_list_cv2 = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in color_list]
def label_on_image(image, label, classes):
    G = np.zeros_like(label)
    B = np.zeros_like(label)
    R = np.zeros_like(label)
    for i in range(1, classes):
        G[label == i] = color_list[i][0] 
        B[label == i] = color_list[i][1] 
        R[label == i] = color_list[i][2] 
    label = np.stack((G, B, R), axis=-1)
    blended_image = 0.1 * image + 0.9 * label
    blended_image[label == 0] = image[label == 0]
    return blended_image


def save_vis(image, pred, label, save_folder_path, name):
    image = image.to(torch.float32).squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) * 255

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    save_temp_label = label_on_image(image, label, classes=args.num_classes)
    save_temp_image = label_on_image(image, pred, classes=args.num_classes)

    # cv2.imwrite(os.path.join(save_folder_path, name + "_image.png"), image)
    # cv2.imwrite(os.path.join(save_folder_path, name + "_label.png"), save_temp_label)
    cv2.imwrite(os.path.join(save_folder_path, name + "_pred.png"), save_temp_image)
    print("")


if __name__ == '__main__':
    inference()
