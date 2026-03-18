import logging


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .subModels.SCINet import SCINet
logger = logging.getLogger(__name__)
from .iou_loss import IOU,DiceLoss,weit_loss,IoULoss,dice_loss_boundary
import cv2

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)


    

def _iou_loss(pred, target):
    # pred = torch.sigmoid(pred)
    pred = torch.softmax(pred, dim=1)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


class Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.funNet = SCINet()

        weights = torch.tensor([0.2,1,1,1,1,1,1,1])
        self.criterionBCE = torch.nn.CrossEntropyLoss(weights)
        self.criterionIOU = _iou_loss
        # self.criterionIOU = IoULoss()
        self.criterionDice = DiceLoss()
        self.criterionCE = weit_loss
        self.criterionDice_boundary = dice_loss_boundary

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def forward(self):
        self.pred = self.funNet(self.input)



    def infer(self, input):
        masks = self.funNet(input)
        # print(masks.shape)
        return masks[0]
    

    def backward_G(self):    

        self.loss_G = 0.5 * self.criterionBCE(self.pred, self.gt_mask)
        self.loss_G += self.criterionDice(self.pred, self.gt_mask)
        # for i in range(1,len(self.pred)):
        #     self.loss_G += 0.5*self.criterionBCE(self.pred[i], self.gt_mask)
        #     self.loss_G += self.criterionDice(self.pred[i], self.gt_mask)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights


