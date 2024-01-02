"""
Training a single model (student or teacher)
"""
from __future__ import print_function

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import time
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger
import random
import pandas as pd
import numpy as np

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.imagenet_dali import get_dali_data_loader
from helper.loops_RFF import train_vanilla as train, validate_vanilla
from learning.contrast_trainer import BaseTrainer
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate, update_dict_to_json
from dataset.histo_dataset import get_histo_dataloader
from model_def import load_model

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # baisc
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=12345, type=int,
                        help='seed for initializing training. choices=[None, 0, 1],')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,60', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # dataset
    # dataset and model
    parser.add_argument('--dataset', type=str, default='prostate_hv', help='dataset')
    parser.add_argument('--model', type=str, default='effiB0',)
    #                         choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
    #                                  'ResNet18', 'ResNet34',
    #                                  'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
    #                                  'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
    #                                  'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50',
    #                                  'effiB0']
    parser.add_argument('--pretrain', type=str, default='PANDA', help='tma_class, tma_kd, ImageNet')
    parser.add_argument('--pre_strict', action='store_false', help='strict by default')

    # Augment
    parser.add_argument('--aug_train', type=str, default='RA', choices=['NULL', 'RA'], help='aug_train')
    parser.add_argument('--crop', type=float, default=0.2, help='crop threshold for RandomResizedCrop')
    parser.add_argument('--image_size', type=int, default=512, help='image_size')
    parser.add_argument('--image_resize', action='store_true')
    parser.add_argument('--n_cls', type=int, default=8, help='image_size')
    parser.add_argument('--skip_test', action='store_true', help='strict by default')

    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)

    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_false', help='Make results reproducible, true by default')
    parser.add_argument('--skip_validation', action='store_false', help='Skip validation of teacher')

    opt = parser.parse_args()

    # set different learning rates for these MobileNet/ShuffleNet models
    # if opt.model_s in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
    #     opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = f'./save/vanilla/Class_ce_{opt.dataset}_StdPre_{opt.pretrain}_strict_{opt.pre_strict}_CPU{opt.num_workers}_GPU{torch.cuda.device_count()}/'
    opt.tb_path = './save/students/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set the model name
    opt.model_name = f'Class_ce_model_{opt.model}_{opt.dataset}_IS_{opt.image_size}_BS{opt.batch_size}_StdPre_{opt.pretrain}_CPU{opt.num_workers}_GPU{torch.cuda.device_count()}_seed{opt.seed}_epoch{opt.epochs}_trial_{opt.trial}'
    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


best_acc = 0
best_f1 = 0
total_time = time.time()


def main():
    opt = parse_option()

    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, best_f1, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)
    opt.rank = 0
    opt.dist_backend = 'nccl'

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    trainer = BaseTrainer(opt)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # if opt.deterministic:
    if opt.seed is not None:
        print('opt.deterministic',  opt.deterministic)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(opt.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)


    print('opt.n_cls:', opt.n_cls)


    model = load_model(opt.model, opt.pretrain, opt.n_cls, opt.pre_strict, opt.gpu, opt.multiprocessing_distributed)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:

                torch.cuda.set_device(opt.gpu)
                model.cuda(opt.gpu)
                DDP = torch.nn.parallel.DistributedDataParallel
                model = DDP(model, device_ids=[opt.gpu])
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion = criterion.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model).cuda()
            else:
                model = model.cuda()

        if not opt.deterministic:
            cudnn.benchmark = True

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            train_loader, val_loader, train_sampler = get_imagenet_dataloader(
                dataset=opt.dataset,
                batch_size=opt.batch_size, num_workers=opt.num_workers,
                multiprocessing_distributed=opt.multiprocessing_distributed)
        else:
            train_loader, val_loader = get_dali_data_loader(opt)
    else:
        train_loader, val_loader, test_loader, train_sampler = get_histo_dataloader(
            opt=opt,
            batch_size=opt.batch_size, num_workers=opt.num_workers,
            multiprocessing_distributed=opt.multiprocessing_distributed)

    # tensorboard
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if not opt.skip_validation:
        # validate teacher accuracy
        teacher_acc,  avg, output_stat = validate_vanilla(test_loader, model, criterion, opt)
        print(output_stat)
        # exit()
        # if opt.dali is not None:
        #     val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')

    # routine
    for epoch in range(1, opt.epochs + 1):
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)
            # No test_sampler because epoch is random seed, not needed in sequential testing.

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, Acc@1 {:.3f}, Time {:.2f}'.format(epoch, train_acc, time2 - time1))

            logger.log_value('lr', optimizer.param_groups[0]['lr'], epoch)
            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        print('GPU %d validating' % (opt.gpu))
        val_acc, val_loss, val_output_stat = validate_vanilla(val_loader, model, criterion, opt, prefix='Val')
        for i in val_output_stat.keys():
            print(i, [val_output_stat[i]])

        if not opt.skip_test:
            print('GPU %d testing' % (opt.gpu))
            test_acc, test_loss, test_output_stat = validate_vanilla(test_loader, model, criterion, opt, prefix='Test')
            for i in test_output_stat.keys():
                print(i, [test_output_stat[i]])


        def f1(a):
            "F1"
            f = 0
            for i in range(a.shape[0]):
                if a[i][i] == 0:
                    f += 0
                else:
                    f += (2 * a[i][i] / a[:, i].sum() * a[i][i] / a[i, :].sum()) / (
                            a[i][i] / a[:, i].sum() + a[i][i] / a[i, :].sum())
            return f

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc_val@1 {:.3f}'.format(val_acc))
            val_f1 = f1(val_output_stat['conf_mat'])
            if not opt.skip_test:
                print(' ** Acc_test@1 {:.3f}'.format(test_acc))

            logger.log_value('val_acc', val_acc, epoch)
            logger.log_value('val_loss', val_loss, epoch)

            if not opt.skip_test:
                logger.log_value('test_acc', test_acc, epoch)
                logger.log_value('test_loss', test_loss, epoch)

            # save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                # Save all
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }

                save_file = os.path.join(opt.save_folder, 'net_best_acc.pth'.format(epoch))
                print('saving the best acc model!')
                torch.save(state, save_file)

            if val_f1 > best_f1:
                best_f1 = val_f1
                # Save all
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict(),
                }

                save_file = os.path.join(opt.save_folder, 'net_best_f1.pth'.format(epoch))
                print('saving the best f1 model!')
                torch.save(state, save_file)

            if not opt.skip_test:
                test_merics = {
                    'val_cf': pd.Series({'conf_mat': val_output_stat['conf_mat']}).to_json(orient='records'),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'test_cf': pd.Series({'conf_mat': test_output_stat['conf_mat']}).to_json(orient='records'),
                    'test_loss': test_loss,
                    'test_acc': test_acc
                }
            else:
                test_merics = {
                    'val_cf': pd.Series({'conf_mat': val_output_stat['conf_mat']}).to_json(orient='records'),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }

            update_dict_to_json(epoch, test_merics, os.path.join(opt.save_folder, "stat.json"))



    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)

        # save parameters
        state = {k: v for k, v in opt._get_kwargs()}

        # No. parameters(M)
        num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
        state['Total params'] = num_params
        state['Total time'] = float('%.2f' % ((time.time() - total_time) / 3600.0))
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(state, params_json_path)


if __name__ == '__main__':
    main()
