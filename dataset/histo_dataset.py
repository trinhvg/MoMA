"""
get data loaders
"""
from __future__ import print_function

import os
import random

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms
from .RandAugment import rand_augment_transform


from .dataset import DatasetSerial, DatasetSerialInstanceSample, DatasetSerial2views


import dataset.histo_list as histo_list
import dataset.histo_list_v1 as histo_list_v1


histodata_list = ['prostate_hv', 'colon_tma_manual', 'panda_512', 'colon_48wsi_v02_s10l08_center_splitv2_2val',
                  'gastric_8class']


def get_data_folder(dataset='imagenet'):
    """
    return the path to store the data
    """
    data_folder = os.path.join('./data', dataset)

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class ImageFolderSample(datasets.ImageFolder):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.k = k
        self.is_sample = is_sample

        print('stage1 finished!')

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                path, target = self.imgs[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_test_loader(dataset='imagenet', batch_size=128, num_workers=8):
    """get the test data loader"""

    if dataset in histo_list:
        data_folder = get_data_folder(dataset)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_folder = os.path.join(data_folder, 'val')
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return test_loader


def get_dataloader_sample(dataset='imagenet', batch_size=128, num_workers=8, is_sample=False, k=4096):
    """Data Loader for ImageNet"""

    if dataset in histo_list:
        data_folder = get_data_folder(dataset)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    # add data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    train_set = ImageFolderSample(train_folder, transform=train_transform, is_sample=is_sample, k=k)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    print('num_samples', len(train_set.samples))
    print('num_class', len(train_set.classes))

    return train_loader, test_loader, len(train_set), len(train_set.classes)


def get_histo_dataloader(opt, batch_size=128, num_workers=16,
                         multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """

    def identity(img, **__):
        return img

    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if opt.dataset == 'prostate_hv':
        resize = transforms.Resize(512)
        if opt.aug_train == 'NULL':
            train_transform = transforms.Compose([
                resize,
                # transforms.RandomResizedCrop(448, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif opt.aug_train == 'RA':
            rgb_mean = (0.485, 0.456, 0.406)
            ra_params = dict(
                translate_const=100,
                img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
            )
            train_transform = transforms.Compose([
                resize,
                # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                       ra_params),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
        val_transform = transforms.Compose([
                resize,
                # transforms.Resize(int(opt.image_size*256/224)),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])
    elif opt.dataset == 'crc_tp':
        if opt.image_size == 224:
            resize = transforms.Resize(224)
            if opt.aug_train == 'NULL':
                train_transform = transforms.Compose([
                    resize,
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif opt.aug_train == 'RA':
                rgb_mean = (0.485, 0.456, 0.406)
                ra_params = dict(
                    translate_const=100,
                    img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
                )
                train_transform = transforms.Compose([
                    resize,
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                           ra_params),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
            val_transform = transforms.Compose([
                resize,
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize, ])

        else:
            if opt.aug_train == 'NULL':
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif opt.aug_train == 'RA':
                rgb_mean = (0.485, 0.456, 0.406)
                ra_params = dict(
                    translate_const=100,
                    img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
                )
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                           ra_params),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
            val_transform = transforms.Compose([
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize, ])

    else:
        if opt.image_resize:
            if opt.aug_train == 'NULL':
                train_transform = transforms.Compose([
                    transforms.Resize(opt.image_size),
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif opt.aug_train == 'RA':
                rgb_mean = (0.485, 0.456, 0.406)
                ra_params = dict(
                    translate_const=100,
                    img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
                )
                train_transform = transforms.Compose([
                    transforms.Resize(opt.image_size),
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                           ra_params),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

            val_transform = transforms.Compose([
                transforms.Resize(opt.image_size),
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize, ])
        else:
            if opt.aug_train == 'NULL':
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif opt.aug_train == 'RA':
                rgb_mean = (0.485, 0.456, 0.406)
                ra_params = dict(
                    translate_const=100,
                    img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
                )
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                           ra_params),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

            val_transform = transforms.Compose([
                    # transforms.Resize(int(opt.image_size*256/224)),
                    # transforms.CenterCrop(opt.image_size),
                    transforms.ToTensor(),
                    normalize,])



    train_pairs, valid_pairs, test_pairs = getattr(histo_list, f'prepare_{opt.dataset}_data')()


    train_dataset = DatasetSerial(
        train_pairs, transform=train_transform)
    val_dataset = DatasetSerial(
        valid_pairs,
        transform=val_transform)
    test_dataset = DatasetSerial(
        test_pairs,
        transform=val_transform)


    # loader
    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=val_sampler)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    print('train images: {}'.format(len(train_dataset)))
    print('val images: {}'.format(len(val_dataset)))
    print('test images: {}'.format(len(test_dataset)))

    return train_loader, val_loader, test_loader, train_sampler


def get_histo_dataloader_2views(opt, batch_size=128, num_workers=16,
                         multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """

    def identity(img, **__):
        return img

    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if opt.dataset == 'prostate_hv':
        resize = transforms.Resize(512)
        if opt.aug_train == 'NULL':
            train_transform = transforms.Compose([
                resize,
                # transforms.RandomResizedCrop(448, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif opt.aug_train == 'RA':
            rgb_mean = (0.485, 0.456, 0.406)
            ra_params = dict(
                translate_const=100,
                img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
            )
            train_transform = transforms.Compose([
                resize,
                # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                       ra_params),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
        val_transform = transforms.Compose([
                resize,
                # transforms.Resize(int(opt.image_size*256/224)),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])
    elif opt.dataset == 'crc_tp':
        if opt.image_size == 224:
            resize = transforms.Resize(224)
            if opt.aug_train == 'NULL':
                train_transform = transforms.Compose([
                    resize,
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif opt.aug_train == 'RA':
                rgb_mean = (0.485, 0.456, 0.406)
                ra_params = dict(
                    translate_const=100,
                    img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
                )
                train_transform = transforms.Compose([
                    resize,
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                           ra_params),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
            val_transform = transforms.Compose([
                resize,
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize, ])

        else:
            if opt.aug_train == 'NULL':
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif opt.aug_train == 'RA':
                rgb_mean = (0.485, 0.456, 0.406)
                ra_params = dict(
                    translate_const=100,
                    img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
                )
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                           ra_params),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
            val_transform = transforms.Compose([
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize, ])

    else:
        if opt.image_resize:
            if opt.aug_train == 'NULL':
                train_transform = transforms.Compose([
                    transforms.Resize(opt.image_size),
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif opt.aug_train == 'RA':
                rgb_mean = (0.485, 0.456, 0.406)
                ra_params = dict(
                    translate_const=100,
                    img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
                )
                train_transform = transforms.Compose([
                    transforms.Resize(opt.image_size),
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                           ra_params),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

            val_transform = transforms.Compose([
                transforms.Resize(opt.image_size),
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize, ])
        else:
            if opt.aug_train == 'NULL':
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif opt.aug_train == 'RA':
                rgb_mean = (0.485, 0.456, 0.406)
                ra_params = dict(
                    translate_const=100,
                    img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
                )
                train_transform = transforms.Compose([
                    # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                           ra_params),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

            val_transform = transforms.Compose([
                    # transforms.Resize(int(opt.image_size*256/224)),
                    # transforms.CenterCrop(opt.image_size),
                    transforms.ToTensor(),
                    normalize,])



    train_pairs, valid_pairs, test_pairs = getattr(histo_list, f'prepare_{opt.dataset}_data')()


    train_dataset = DatasetSerial2views(
        train_pairs, transform=train_transform, two_crop=opt.two_crop)
    val_dataset = DatasetSerial2views(
        valid_pairs,
        transform=val_transform)
    test_dataset = DatasetSerial2views(
        test_pairs,
        transform=val_transform)


    # loader
    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=val_sampler)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    print('train images: {}'.format(len(train_dataset)))
    print('val images: {}'.format(len(val_dataset)))
    print('test images: {}'.format(len(test_dataset)))

    return train_loader, val_loader, test_loader, train_sampler


def get_histo_testloader(opt, batch_size=128, num_workers=16,
                         multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """



    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.image_resize:
        if opt.aug_train == 'RA_375_512':
            val_transform = transforms.Compose([
                transforms.CenterCrop(375),
                transforms.Resize(512),
                transforms.ToTensor(),
                normalize, ])
        else:
            val_transform = transforms.Compose([
                transforms.Resize(opt.image_size),
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize, ])
    else:
        val_transform = transforms.Compose([
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])



    test_pairs = getattr(histo_list, f'prepare_{opt.dataset}_test_data')()

    random.shuffle(test_pairs)
    test_dataset = DatasetSerial(
        test_pairs,
        transform=val_transform)


    # loader
    if multiprocessing_distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = None


    # test_loader = DataLoader(test_dataset,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          num_workers=num_workers,
    #                          pin_memory=True,
    #                          sampler=test_sampler)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    print('test images: {}'.format(len(test_dataset)))

    return test_loader



def get_histo_testloader_2(dataset, opt, batch_size=128, num_workers=16,
                         multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """



    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.image_resize:
        val_transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            # transforms.Resize(int(opt.image_size*256/224)),
            # transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            normalize, ])
    else:
        val_transform = transforms.Compose([
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])



    test_pairs = getattr(histo_list, f'prepare_{dataset}_test_data')()


    # test_pairs_0 = [pair for pair in test_pairs if pair[1] ==0]
    # test_pairs_1 = [pair for pair in test_pairs if pair[1] ==1]
    # test_pairs_2 = [pair for pair in test_pairs if pair[1] ==2]
    # test_pairs_3 = [pair for pair in test_pairs if pair[1] ==3]
    # test_pairs = test_pairs_0[:127] + test_pairs_1[:127] + test_pairs_2[:127] + test_pairs_3[:127]

    test_dataset = DatasetSerial(
        test_pairs,
        transform=val_transform)


    # loader
    if multiprocessing_distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = None


    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    print('test images: {}'.format(len(test_dataset)))

    return test_loader




def get_histo_viz_testloader(opt, batch_size=128, num_workers=16,
                         multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """



    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.image_resize:
        val_transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            # transforms.Resize(int(opt.image_size*256/224)),
            # transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            normalize, ])
    else:
        val_transform = transforms.Compose([
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])



    test_pairs = getattr(histo_list, f'prepare_{opt.dataset}_test_data')()
    test_pairs_viz = []
    for i in range(opt.n_cls):
        test_pairs_viz += [pair for pair in test_pairs if pair[1] ==i][:opt.num_per_class]
        # test_pairs_1 = [pair for pair in test_pairs if pair[1] ==1]
        # test_pairs_2 = [pair for pair in test_pairs if pair[1] ==2]
        # test_pairs_3 = [pair for pair in test_pairs if pair[1] ==3]
        # test_pairs = test_pairs_0[:16] + test_pairs_1[:16] + test_pairs_2[:16] + test_pairs_3[:16]
    test_dataset = DatasetSerial(
        test_pairs_viz,
        transform=val_transform)


    # loader
    if multiprocessing_distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = None


    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    print('test images: {}'.format(len(test_dataset)))

    return test_loader





def get_histo_5fold_testloader(opt, dataset='colon_tma', batch_size=128, num_workers=16,
                         is_sample=False, k=4096, multiprocessing_distributed=False,fold_idx=0):
    """
    Data Loader for imagenet
    """


    opt.crop = 0.4
    # opt.image_size = 448

    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.aug_train == 'RA05':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.aug_train == 'RA':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

    if opt.aug_train == 'RA05':
        val_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            normalize, ])
    else:
        val_transform = transforms.Compose([
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            normalize, ])


    train_pairs, valid_pairs = \
        getattr(histo_list_v1, f'prepare_{opt.dataset}_data')(subdir=opt.subdir, subdir_3=opt.subdir_3, fold_idx=fold_idx)


    if opt.subdir != '1':
        valid_pairs = train_pairs + valid_pairs
    val_dataset = DatasetSerial(
        valid_pairs,
        img_transform=val_transform)

    # loader
    if multiprocessing_distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None


    val_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=val_sampler)


    print('val images: {}'.format(len(val_dataset)))

    return val_loader


def get_histo_viz_5fold_testloader(opt, dataset='colon_tma', batch_size=128, num_workers=16,
                         is_sample=False, k=4096, multiprocessing_distributed=False,fold_idx=0):
    """
    Data Loader for imagenet
    """


    opt.crop = 0.4
    # opt.image_size = 448

    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.aug_train == 'RA05':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.aug_train == 'RA':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))

    if opt.aug_train == 'RA05':
        val_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            normalize, ])
    else:
        val_transform = transforms.Compose([
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            normalize, ])


    train_pairs, valid_pairs = \
        getattr(histo_list_v1, f'prepare_{opt.dataset}_data')(subdir=opt.subdir, subdir_3=opt.subdir_3, fold_idx=fold_idx)


    if opt.subdir != '1':
        valid_pairs = train_pairs + valid_pairs

    test_pairs_viz = []
    for i in range(opt.n_cls):
        test_pairs_viz += [pair for pair in valid_pairs if pair[1] ==i][:opt.num_per_class]


    val_dataset = DatasetSerial(
        test_pairs_viz,
        transform=val_transform)

    # loader
    if multiprocessing_distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None


    val_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=val_sampler)


    print('val images: {}'.format(len(val_dataset)))

    return val_loader



def get_histo_dataloader_448(opt, dataset='colon_tma', batch_size=128, num_workers=16,
                         multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """
    opt.crop = 0.5
    opt.image_size = 448

    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if opt.dataset == 'prostate_hv':
        resize = transforms.Resize(512)
        if opt.aug_train == 'NULL':
            train_transform = transforms.Compose([
                resize,
                transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif opt.aug_train == 'RA':
            rgb_mean = (0.485, 0.456, 0.406)
            ra_params = dict(
                translate_const=100,
                img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
            )
            train_transform = transforms.Compose([
                resize,
                transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                       ra_params),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
        val_transform = transforms.Compose([
                resize,
                # transforms.Resize(int(opt.image_size*256/224)),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])
    else:
        if opt.aug_train == 'NULL':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif opt.aug_train == 'RA':
            rgb_mean = (0.485, 0.456, 0.406)
            ra_params = dict(
                translate_const=100,
                img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
            )
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                       ra_params),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
        val_transform = transforms.Compose([
                # transforms.Resize(int(opt.image_size*256/224)),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])



    train_pairs, valid_pairs, test_pairs = getattr(histo_list, f'prepare_{opt.dataset}_data')()


    train_dataset = DatasetSerial(
        train_pairs, transform=train_transform)
    val_dataset = DatasetSerial(
        valid_pairs,
        transform=val_transform)
    test_dataset = DatasetSerial(
        test_pairs,
        transform=val_transform)


    # loader
    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=val_sampler)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    print('train images: {}'.format(len(train_dataset)))
    print('val images: {}'.format(len(val_dataset)))
    print('test images: {}'.format(len(test_dataset)))

    return train_loader, val_loader, test_loader, train_sampler

def get_histo_dataloader_sample(opt,  dataset='colon_tma', batch_size=128, num_workers=8, k=4096,  mode='exact',
                         is_sample=True, percent=1.0, multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """


    # transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if opt.dataset == 'prostate_hv':
        resize = transforms.Resize(512)
        if opt.aug_train == 'NULL':
            train_transform = transforms.Compose([
                resize,
                # transforms.RandomResizedCrop(448, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif opt.aug_train == 'RA':
            rgb_mean = (0.485, 0.456, 0.406)
            ra_params = dict(
                translate_const=100,
                img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
            )
            train_transform = transforms.Compose([
                resize,
                # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                       ra_params),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
        val_transform = transforms.Compose([
                resize,
                # transforms.Resize(int(opt.image_size*256/224)),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])
    else:
        if opt.aug_train == 'NULL':
            train_transform = transforms.Compose([
                # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif opt.aug_train == 'RA':
            rgb_mean = (0.485, 0.456, 0.406)
            ra_params = dict(
                translate_const=100,
                img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
            )
            train_transform = transforms.Compose([
                # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                       ra_params),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplementedError('aug not found: {}'.format(opt.aug_linear))
        val_transform = transforms.Compose([
                # transforms.Resize(int(opt.image_size*256/224)),
                # transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                normalize,])



    train_pairs, valid_pairs, test_pairs = getattr(histo_list, f'prepare_{opt.dataset}_data')()


    train_dataset = DatasetSerialInstanceSample(
        train_pairs,
        train=True,
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent
    )
    val_dataset = DatasetSerial(
        valid_pairs,
        transform=val_transform)
    test_dataset = DatasetSerial(
        test_pairs,
        transform=val_transform)


    # loader
    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    n_data = len(train_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=val_sampler)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             sampler=test_sampler)

    print('train images: {}'.format(len(train_dataset)))
    print('val images: {}'.format(len(val_dataset)))
    print('test images: {}'.format(len(test_dataset)))

    return train_loader, val_loader, test_loader, train_sampler, n_data