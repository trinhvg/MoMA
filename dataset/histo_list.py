import os
import csv
import glob
import random
from collections import Counter

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.utils import make_grid
from imgaug import augmenters as iaa
import pandas as pd
import h5py


#############
def print_number_of_sample(train_set, valid_set, test_set):
    train_label = [train_set[i][1] for i in range(len(train_set))]
    print("train", Counter(train_label))
    print(len(train_set))
    valid_label = [valid_set[i][1] for i in range(len(valid_set))]
    print("valid", Counter(valid_label))
    test_label = [test_set[i][1] for i in range(len(test_set))]
    print("test", Counter(test_label))
    return 0


####

def prepare_panda_512_data(fold_idx=0):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        if parse_label:
            label_list = [int(file_path.split('_')[-3]) - 2 for file_path in file_list]
        else:
            label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    data_root_dir = './PANDA_RA_patch'
    train_set_1 = load_data_info('%s/1*/*.png' % data_root_dir)
    train_set_2 = load_data_info('%s/2*/*.png' % data_root_dir)
    train_set_3 = load_data_info('%s/3*/*.png' % data_root_dir)
    train_set_4 = load_data_info('%s/4*/*.png' % data_root_dir)
    train_set_5 = load_data_info('%s/5*/*.png' % data_root_dir)
    train_set_6 = load_data_info('%s/6*/*.png' % data_root_dir)

    train_set = train_set_1 + train_set_2 + train_set_4 + train_set_6
    valid_set = train_set_3
    test_set = train_set_5
    print_number_of_sample(train_set, valid_set, test_set)
    print(len(train_set + valid_set + test_set))
    return train_set, valid_set, test_set
# prepare_panda_512_data()


def prepare_prostate_hv_data(fold_idx=0):
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        # print(Counter(label_list))
        return list(zip(file_list, label_list))


    data_root_dir = './patch_data/prostate_HV_patch_750/'
    data_root_dir_train = f'{data_root_dir}/patches_train_750_v0/'
    data_root_dir_val = f'{data_root_dir}/patches_validation_750_v0/'
    data_root_dir_test = f'{data_root_dir}/patches_test_750_v0/'

    train_set_111 = load_data_info('%s/ZT111*/*.jpg' % data_root_dir_train)
    train_set_199 = load_data_info('%s/ZT199*/*.jpg' % data_root_dir_train)
    train_set_204 = load_data_info('%s/ZT204*/*.jpg' % data_root_dir_train)
    valid_set = load_data_info('%s/ZT76*/*.jpg' % data_root_dir_val)
    test_set = load_data_info('%s/patho_1/*/*.jpg' % data_root_dir_test)
    test_set_2 = load_data_info('%s/patho_2/*/*.jpg' % data_root_dir_test)

    train_set = train_set_111 + train_set_199 + train_set_204
    print_number_of_sample(train_set, valid_set, test_set)
    print_number_of_sample(train_set, valid_set, test_set_2)
    return train_set, valid_set, test_set








def prepare_kather19_data():
    def load_data_info( pathname, covert_dict):
        file_list = glob.glob(pathname)
        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    const_kather19 = {
        'ADI': ('ADI', 0), 'BACK': ('BACK', 1), 'DEB': ('DEB', 2), 'LYM': ('LYM', 3), 'MUC': ('MUC', 4),
        'MUS': ('MUS', 5), 'NORM': ('NORM', 6), 'STR': ('STR', 7), 'TUM': ('TUM', 8)
    }

    k19_path= './colon_class/NCT-CRC-HE-100K/'
    k19_val_path = './colon_class/CRC-VAL-HE-7K/'
    train_set = load_data_info(pathname = f'{k19_path}/*/*.tif', covert_dict=const_kather19)
    val_set = load_data_info(pathname = f'{k19_val_path}/*/*.tif', covert_dict=const_kather19)

    return train_set, val_set, []


def prepare_kather19_nonorm_data():
    def load_data_info( pathname, covert_dict):
        file_list = glob.glob(pathname)
        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    const_kather19 = {
        'ADI': ('ADI', 0), 'BACK': ('BACK', 1), 'DEB': ('DEB', 2), 'LYM': ('LYM', 3), 'MUC': ('MUC', 4),
        'MUS': ('MUS', 5), 'NORM': ('NORM', 6), 'STR': ('STR', 7), 'TUM': ('TUM', 8)
    }

    k19_path= './colon_class/NCT-CRC-HE-100K-NONORM/'
    k19_val_path = './colon_class/CRC-VAL-HE-7K/'
    train_set = load_data_info(pathname = f'{k19_path}/*/*.tif', covert_dict=const_kather19)
    val_set = load_data_info(pathname = f'{k19_val_path}/*/*.tif', covert_dict=const_kather19)

    return train_set, val_set, []


def prepare_kather19_test_data():
    def load_data_info( pathname, covert_dict):
        file_list = glob.glob(pathname)
        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    const_kather19 = {
        'ADI': ('ADI', 0), 'BACK': ('BACK', 1), 'DEB': ('DEB', 2), 'LYM': ('LYM', 3), 'MUC': ('MUC', 4),
        'MUS': ('MUS', 5), 'NORM': ('NORM', 6), 'STR': ('STR', 7), 'TUM': ('TUM', 8)
    }

    k19_val_path = './colon_class/CRC-VAL-HE-7K/'
    val_set = load_data_info(pathname = f'{k19_val_path}/*/*.tif', covert_dict=const_kather19)

    return val_set

def prepare_kather16_test_data():
    def load_data_info( pathname, covert_dict):
        file_list = glob.glob(pathname)
        COMPLEX_list = glob.glob(f'{k16_path}/03_COMPLEX/*.tif')
        file_list = [elem for elem in file_list if elem not in COMPLEX_list]
        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))
    const_kather16 = {
        '07_ADIPOSE': ('07_ADIPOSE', 0), '08_EMPTY': ('08_EMPTY', 1), '05_DEBRIS': ('05_DEBRIS', 2),
        '04_LYMPHO': ('04_LYMPHO', 3), '06_MUCOSA': ('06_MUCOSA', 6), '02_STROMA': ('02_STROMA', 7),
        '01_TUMOR': ('01_TUMOR', 8)
    }

    k16_path= './colon_class/Kather_texture_2016_image_tiles_5000'
    val_set = load_data_info(pathname = f'{k16_path}/*/*.tif', covert_dict=const_kather16)

    return val_set


def prepare_prostate_ubc_test_data(fold_idx=0):
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [convert_dict[int(file_path.split('_')[-1].split('.')[0])] for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    convert_dict = {0:0, 2:1, 3:2, 4:3}
    data_root_dir_test = f'./patch_data/prostate_miccai_2019_patches_690_80_step05_test'
    test_set = load_data_info('%s/*/*.jpg' % data_root_dir_test)

    return test_set
