import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class WaymoFrameDataset(Dataset):

    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return max(0, len(self.sequence) - 1)

    def __getitem__(self, idx):
        ts_a, img_a = self.sequence[idx]
        ts_b, img_b = self.sequence[idx + 1]
        return {
            "frame_a": img_a,
            "frame_b": img_b,
            "ts_a": ts_a,
            "ts_b": ts_b,
        }


class CropDataset(Dataset):

    DINOV2_MEAN = [0.485, 0.456, 0.406]
    DINOV2_STD = [0.229, 0.224, 0.225]

    def __init__(self, crops, input_size=224):
        self.crops = crops
        self.transform = T.Compose([
            T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=self.DINOV2_MEAN, std=self.DINOV2_STD),
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        img = Image.fromarray(self.crops[idx])
        return self.transform(img)


class AugmentedCropDataset(Dataset):

    DINOV2_MEAN = [0.485, 0.456, 0.406]
    DINOV2_STD = [0.229, 0.224, 0.225]

    def __init__(self, crops, input_size=224):
        self.crops = crops
        self.augment = T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=self.DINOV2_MEAN, std=self.DINOV2_STD),
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        img = Image.fromarray(self.crops[idx])
        view1 = self.augment(img)
        view2 = self.augment(img)
        return view1, view2
