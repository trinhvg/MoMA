from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
from torchvision import datasets
from PIL import Image
import cv2
from torchvision import transforms, datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DatasetSerial(data.Dataset):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, pair_list, transform=None, target_transform=None):
        self.pair_list = pair_list
        self.transform = transform
        self.target_transform = target_transform
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.pair_list[index]
        image = pil_loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image

        return img, target



    def __len__(self):
        return len(self.pair_list)

class DatasetSerial2views(data.Dataset):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, pair_list, transform=None, target_transform=None,  two_crop=False):
        self.pair_list = pair_list
        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.pair_list[index]
        image = pil_loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        return img, target



    def __len__(self):
        return len(self.pair_list)



class DatasetSerialInstanceSample(DatasetSerial):
    """
    CIFAR100Instance+Sample Dataset
    """

    def __init__(self,  pair_list,  train=True, transform=None, target_transform=None,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(pair_list=pair_list, transform=transform, target_transform=target_transform)
        self.k = k
        self.pair_list = pair_list
        self.mode = mode
        self.is_sample = is_sample

        num_samples = len(self.pair_list)
        label = [pair[1] for pair in self.pair_list]
        num_classes = len(set(label))

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive, dtype=object)
        self.cls_negative = np.asarray(self.cls_negative, dtype=object)

    def __getitem__(self, index):

        path, target = self.pair_list[index]
        img = pil_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
