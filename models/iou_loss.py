import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        # pred = torch.softmax(pred, dim=1)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)
    
class IoULoss(nn.Module):
    def __init__(self, n_classes=8):
        super(IoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = torch.sum(pred * target)
        union = torch.sum(pred + target) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target.unsqueeze(1))
        # if weight is None:
        #     weight = [1] * self.n_classes
        weight = [1,1,5,1,1,1,1,1]
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._iou(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            class_wise_dice.append(dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes=8):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss


    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target.unsqueeze(1))
        # if weight is None:
        #     weight = [1] * self.n_classes
        # weight = [1,1,0.5,1,1,5,1,5]
        weight = [1,1,1,1,1,1,1,1]
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(1, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / (self.n_classes-1)
    
# def weit_loss(pred, mask):
#     weit = torch.abs(torch.softmax(pred,dim=1) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wbce + wiou).mean()

def weit_loss(pred, mask):
    weit = torch.abs(torch.softmax(pred,dim=1) - mask)
    target = torch.argmax(mask,dim=1)
    cls_weit = torch.tensor([1,1,5,1,1,1,1,1],dtype=torch.float).cuda()
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    wbce = torch.sum(wbce * cls_weit, dim=0) / torch.sum(cls_weit)
    return wbce.mean()
    # pred = torch.softmax(pred,dim=1)
    # inter = ((pred * mask)*weit).sum(dim=(2, 3))
    # union = ((pred + mask)*weit).sum(dim=(2, 3))
    # wiou = 1 - (inter + 1)/(union - inter+1)
    # return (wbce + wiou).mean()

# def weit_loss(pred, mask):
#     pixel_weit = torch.abs(torch.sigmoid(pred) - mask)
#     # weit = 1 + 5*torch.abs(F.avg_pool2d(weit, kernel_size=31, stride=1, padding=15) - mask)
#     class_weit = [1,1,5,1,1,1,1,1]
#     num_class = pred.shape[1]
#     loss = 0
#     for cls in range(0, num_class):
#         # cls_pred = pred[:, cls, ...].unsqueeze(1)
#         # # cls_mask = (mask == cls).float().unsqueeze(1)
#         # cls_mask = mask[:, cls, ...].unsqueeze(1)
#         cls_pred = pred[:, cls, ...]
#         cls_mask = mask[:, cls, ...]
#         # CE
#         wbce = F.binary_cross_entropy_with_logits(cls_pred, cls_mask, reduction='none')
#         # wbce = (pixel_weit[:, cls, ...].unsqueeze(1)*wbce).sum(dim=(2, 3)) / pixel_weit[:, cls, ...].unsqueeze(1).sum(dim=(2, 3))
#         wbce = (pixel_weit[:, cls, ...]*wbce).sum(dim=(1, 2)) / pixel_weit[:, cls, ...].sum(dim=(1, 2))
#         # IOU
#         # cls_pred = torch.sigmoid(cls_pred)
#         # inter = ((cls_pred * cls_mask)*pixel_weit[:, cls, ...]).sum(dim=(1, 2))
#         # union = ((cls_pred + cls_mask)*pixel_weit[:, cls, ...]).sum(dim=(1, 2))
#         # wiou = 1 - (inter + 1e-5)/(union - inter + 1e-5)
#         # # Dice
#         # smooth = 1e-10
#         # intersect = ((cls_pred * cls_mask * weit)).sum(dim=(2, 3))
#         # y_sum = (cls_mask * weit).sum(dim=(2, 3))
#         # z_sum = (cls_pred * weit).sum(dim=(2, 3))
#         # wdice = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         # if wdice.mean(dim=0) < 0:
#         #     print(wdice.mean(dim=0))

#         # loss = loss + (wbce + wiou).mean() * class_weit[cls]
#         loss = loss + (wbce).mean() * class_weit[cls]
#     loss = loss / num_class
#     return loss
def dice_loss_boundary(pred, target):
    smooth = 1e-10
    intersection = torch.sum(pred * target)
    return 1 - (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)