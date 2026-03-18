import os
import h5py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class build_Dataset(Dataset):
    def __init__(self, args, data_dir, split, transform=None, model="None"):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.model = model
        self.args = args
        if data_dir == 'endovis18':
            if self.split == "train":
                image_path = os.path.join("/opt/data/private/data/endovis18/train/images")
            elif self.split == "val":
                image_path = os.path.join("/opt/data/private/data/endovis18/val/images")
            else:
                image_path = os.path.join("/opt/data/private/data/endovis18/val/images")
            sample_list = os.listdir(image_path)
            sample_list = [os.path.join(image_path, item) for item in sample_list]
            self.sample_list = sample_list
            print("train total {} samples".format(len(self.sample_list)))
        else:
            if self.split == "train":
                image_path = ['/opt/data/private/data/organize_endovis2017/fold2/images',
                            '/opt/data/private/data/organize_endovis2017/fold3/images','/opt/data/private/data/organize_endovis2017/fold1/images']
            elif self.split == "val":
                image_path = ['/opt/data/private/data/organize_endovis2017/fold0/images']
            else:
                image_path = ['/opt/data/private/data/endovis2017/endovis17/instrument_2017_test/images']
            self.sample_list = []
            for i in range(len(image_path)):
                sample_list = os.listdir(image_path[i])
                sample_list = [os.path.join(image_path[i], item) for item in sample_list]
                self.sample_list += sample_list
            print("train total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def one_hot_encoder(self, input_tensor, n_classes):
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=0)
        return output_tensor.float()

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = cv2.imread(case) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        label_path = case.replace("images", "annotations")
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = image.astype(np.float32)
            if self.split == "test" and self.data_dir == 'endovis17':
                height, width = 1024, 1280
                h_start, w_start = 28, 320
                image = image[h_start: h_start + height, w_start: w_start + width]
                label = label[h_start: h_start + height, w_start: w_start + width]
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
        image, label = torch.tensor(image.transpose(2, 0, 1).astype('float32')), torch.tensor(label)
        label = self.one_hot_encoder(label.unsqueeze(0), self.args.num_classes)

        file_name, _ = os.path.splitext(os.path.basename(case))
        sample = {"inp": image, "gt": label, "name": file_name}
        return sample






