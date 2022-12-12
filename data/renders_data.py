import os
import logging
import numpy as np

from scipy.spatial.transform import Rotation as R
from torchvision.transforms import ToTensor
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

import sys
import os

from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader

def load_metadatas(path):
    with open(path / 'metadatas.json', 'r') as f:
        metadatas = json.load(f)
    for metadata_list in metadatas:
        for metadata in metadata_list:
            metadata['path'] = path / metadata['asset_id']
    return metadatas

def load_transforms(metadata):
    with open(metadata['path'] / 'transforms.json', 'r') as f:
        transforms = json.load(f)
    return transforms

def load_render(metadata, channel='rgba'):
    return Image.open(metadata['path'] / metadata[f'{channel}_path'])

class RenderCoupleDataset(Dataset):
    def __init__(self, data, data_transforms, data_augmentations=None):
        self.data = data
        self.data_transforms = data_transforms
        self.data_augmentations = data_augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, target_data = self.data[idx]
        input_image = load_render(input_data)
        target_image = load_render(target_data)
        input_image = transforms.ToTensor()(input_image)
        target_image = transforms.ToTensor()(target_image)
        if self.data_augmentations is not None:
            input_image, target_image = self.data_augmentations(input_image, target_image)
        if self.transforms is not None:
            input_image = self.data_transforms(input_image)
            target_image = self.data_transforms(target_image)
        return input_image, target_image
    
class RenderDataset(Dataset):
    def __init__(self, data, data_transforms=None, data_augmentations=None):
        self.data = data
        self.data_transforms = data_transforms
        self.data_augmentations = data_augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = load_render(self.data[idx])
        image = transforms.ToTensor()(image)
        if self.data_augmentations is not None:
            image = self.data_augmentations(image)
        if self.transforms is not None:
            image = self.data_transforms(image)
        return image
    
def makeDataLoader(data, config, dataset_class=RenderDataset, data_transforms=None, data_augmentations=None, test_size=0.1):
    train_ids, valid_ids = train_test_split(data, test_size=test_size, random_state=config.seed)
    train_dataset = dataset_class(data, data_transforms, data_augmentations)
    valid_dataset = dataset_class(data, data_transforms)
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.eval_batch_size, shuffle=False, **kwargs)
    
    train_size = len(train_dataset)
    valid_size = len(valid_dataset)
    
    return train_loader, valid_loader, train_size, valid_size

class BackgroundColor(object):
    def __init__(self, color):
        self.color = color

    def __call__(self, image):
        rgb = image[:3]
        alpha = image[3]
        bg_color = self.color.repeat(1, *image.shape[-2:])
        blended_rgb = torch.mul(rgb, alpha) + torch.mul(bg_color, 1 - alpha)
        return blended_rgb

class RandomHue(object):
    def __init__(self):
        pass
    
    def adjust_hue(self, image, hue):
        rgb = image[:3]
        alpha = image[3:]
        rgb = transforms.functional.adjust_hue(rgb, hue)
        image = torch.cat([rgb, alpha], 0)
        return image
    
    def __call__(self, data, target):
        hue = np.random.rand() - 0.5
        data = self.adjust_hue(data, hue)
        target = self.adjust_hue(target, hue)
        return data, target

NormTorchToPil = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ],
                         std = [ 1/0.5 ]),
    transforms.Normalize(mean = [ -0.5 ],
                         std = [ 1., 1., 1. ]),
    transforms.ToPILImage(),
])

def plot_predictions(data, output, target):
    fig = plt.figure(figsize=(2*len(data), 2*3))
    for idx in np.arange(len(target)):
        ax = fig.add_subplot(3, len(data), idx+1, xticks=[], yticks=[])
        plt.imshow(get_image(data[idx]))
        ax = fig.add_subplot(3, len(data), idx+1+len(data), xticks=[], yticks=[])
        plt.imshow(get_image(output[idx]))
        ax = fig.add_subplot(3, len(data), idx+1+len(data)*2, xticks=[], yticks=[])
        plt.imshow(get_image(target[idx]))
    return fig