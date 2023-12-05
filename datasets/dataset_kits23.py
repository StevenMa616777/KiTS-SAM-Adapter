import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
from PIL import Image


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def intensity_to_label(label_img):
    label = (label_img / 255. * 3).astype(np.int)
    return label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label_img = sample['image'], sample['label']
        label_img = label_img[:,:,0]
        label = intensity_to_label(label_img)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # CxHxW
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


class KiTS23_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            filename = self.sample_list[idx].strip('\n')

            # Load the image and label
            image_path = os.path.join(self.data_dir, "image_slices", filename+'.png')
            label_path = os.path.join(self.data_dir, "label_slices", filename+'.png')
            image = np.array(Image.open(image_path))
            label = np.array(Image.open(label_path))
        else:
            case_name = self.sample_list[idx].strip('\n')
            filename = case_name
            file_path = os.path.join(self.data_dir, f"{case_name}.h5")
            with h5py.File(file_path, 'r') as f:
                image = f['image'][:]
                label = f['label'][:]

            # 将图像和标签维度重新排列到合适的形状，例如将其从 (D, H, W, C) 转换为 (C, D, H, W)
            image = np.transpose(image, (3, 0, 1, 2))
            label = np.transpose(label, (3, 0, 1, 2))

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = filename
        return sample
