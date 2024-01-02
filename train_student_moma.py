"""
DDP training for Contrastive Learning
"""

from __future__ import print_function

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import re
import argparse
import time
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger
import random
import pandas as pd
import numpy as np

from models import model_dict
from models.util import ConvReg, SelfA, SRRL, SimKD

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader, get_dataloader_sample
from dataset.imagenet_dali import get_dali_data_loader

from helper.loops_moma import validate_vanilla, validate_distill, train_distill_moma
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate, update_dict_to_json

from crd.criterion import CRDLoss
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, VIDLoss, SemCKDLoss

from dataset.histo_dataset import get_histo_dataloader, get_histo_dataloader_sample
from learning.contrast_trainer import ContrastTrainer
from MoMA.mem_moco import build_mem
from MoMA.criterion_moco_att import CMO

from model_def import load_model

split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=12345, type=int,
                        help='seed for initializing training. choices=[None, 0, 1],')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,60', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='prostate_hv', help='dataset')
    parser.add_argument('--model_s', type=str, default='effiB0',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'ResNet18', 'ResNet34',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50',
                                 'effiB0'])
    parser.add_argument('--model_t', type=str, default='effiB0')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # Augment
    parser.add_argument('--aug_train', type=str, default='RA', choices=['NULL', 'RA'], help='aug_train')
    parser.add_argument('--crop', type=float, default=0.2, help='crop threshold for RandomResizedCrop')
    parser.add_argument('--image_size', type=int, default=512, help='image_size')
    parser.add_argument('--image_resize', action='store_true')
    parser.add_argument('--n_cls', type=int, default=8, help='image_size')
    parser.add_argument('--skip_test', action='store_true', help='strict by default')

    # distillation
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd')
    # choices=['kd', 'hint', 'attention', 'similarity', 'vid', 'crd', 'semckd','srrl', 'simkd', 'moma'])
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('-f', '--factor', type=int, default=2, help='factor size of SimKD')
    parser.add_argument('-s', '--soft', type=float, default=1.0, help='attention scale of SemCKD')

    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])

    # NCE distillation
    parser.add_argument('--feat_dim', default=512, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--alpha', default=0.999, type=float,
                        help='momentum coefficients for moco encoder update')
    parser.add_argument('--mem', default='MoCo', type=str,
                        choices=['MoCo', 'MoCoST', 'MoCoSSTT'])
    parser.add_argument('--head', default='None', type=str,
                        choices=['None', 'linear', 'mlp'])

    # Distill option
    parser.add_argument('--weight', type=float, default=1e-4, help='number')
    parser.add_argument('--std_pre', type=str, default='PANDA', help='tma_class, tma_kd, ImageNet')
    parser.add_argument('--std_strict', action='store_false', help='strict by default')
    parser.add_argument('--tec_pre', type=str, default='ImageNet', help='tma_class, tma_kd, ImageNet')
    parser.add_argument('--tec_strict', action='store_false', help='strict by default')
    parser.add_argument('--attn', type=str, default='self')

    # multiprocessing
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
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

    if opt.distill == 'moma':
        opt.nce_t = 0.15

    # set the path of model and tensorboard
    opt.model_path = f'./save/kd_{opt.dataset}_{opt.model_s}_StdPre_{opt.std_pre}_and_TecPre_{opt.tec_pre}_CPU{opt.num_workers}_GPU{torch.cuda.device_count()}/'
    opt.tb_path = './save/students/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_r', '{}_a', '{}_b', '{}_{}'])
    opt.model_name = model_name_template.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                opt.cls, opt.div, opt.beta, opt.trial)

    print(opt.model_name)

    opt.model_name = f'{opt.distill}_{opt.dataset}_{opt.model_s}_BS{opt.batch_size}_lr_{opt.learning_rate}_decay' \
                     f'_{opt.weight_decay}_seed{opt.seed}_imageS_{opt.image_size}_cosine_{opt.cosine}' \
                     f'_StdPre_{opt.std_pre}_strict_{opt.std_strict}_and_TecPre_{opt.tec_pre}_strict_{opt.tec_strict}_TB0_SB0_BZ64_attn_{opt.attn}'

    if opt.distill == 'moma':
        opt.model_name = f'{opt.model_name}_{opt.mem}_head_{opt.head}_{opt.feat_dim}'

    opt.model_name = f'{opt.model_name}_c{opt.cls}_d{opt.div}_b{opt.beta}_trial_{opt.trial}'

    print('opt.model_path: ', opt.model_path)
    print('opt.model_name: ', opt.model_name)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model


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

    trainer = ContrastTrainer(opt)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # if opt.deterministic:
    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(opt.seed)

    # model
    n_cls = {
        'cifar100': 100,
        'imagenet': 1000,
        'colon_tma_manual': 4,
        'panda_512': 4,
        'prostate_hv': 4,
        'gastric': 8,
        'gastric_cancer_ano0810_bright230_8class_wsi_downsample': 8,
        'gastric_cancer_ano0805_bright230_8class_wsi_downsample': 8,
        'gastric_cancer_tma_sv0': 8,
    }.get(opt.dataset, None)

    print('opt.n_cls: ', opt.n_cls)

    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)
    else:
        data = torch.randn(2, 3, 512, 512)

    model_s = load_model(opt.model_s, opt.std_pre, opt.n_cls, opt.std_strict, opt.gpu, opt.multiprocessing_distributed)

    model_t = load_model(opt.model_t, opt.tec_pre, opt.n_cls, opt.tec_strict, opt.gpu, opt.multiprocessing_distributed)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'crd':
        "Contrastive Representation Distillation"
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        if opt.dataset == 'cifar100':
            opt.n_data = 50000
        if opt.dataset == 'colon_tma_manual':
            opt.n_data = 7027
        if opt.dataset == 'cifar100':
            opt.n_data = 15303
        else:
            opt.n_data = 1281167

        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'moma':
        "Contrastive Representation Distillation with momentum contrastive learning"

        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]

        #############################################################################################################
        # This part for CMO
        if opt.head == 'None':
            opt.feat_dim = opt.s_dim
        contrast = build_mem(opt)
        contrast.cuda()
        # optional step: synchronize memory
        trainer.broadcast_memory(contrast)
        #############################################################################################################
        criterion_kd = CMO(opt)
        if opt.head == 'mlp':
            module_list.append(criterion_kd.embed_s)
            module_list.append(criterion_kd.embed_t)
            trainable_list.append(criterion_kd.embed_s)
            criterion_kd.embed_t.eval()

        if opt.attn == 'self_mix':
            trainable_list.append(criterion_kd.atts)
        elif opt.attn == 'dual':
            trainable_list.append(criterion_kd.atts_p)
            trainable_list.append(criterion_kd.atts_n)
        elif opt.attn == 'self_nomix':
            trainable_list.append(criterion_kd.atts_q)
            trainable_list.append(criterion_kd.atts_k)
        else:
            trainable_list.append(criterion_kd.atts_q)
            trainable_list.append(criterion_kd.atts_k)
            trainable_list.append(criterion_kd.atts_queue)

    elif opt.distill == 'semckd':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = SemCKDLoss()
        self_attention = SelfA(opt.batch_size, s_n, t_n, opt.soft)
        module_list.append(self_attention)
        trainable_list.append(self_attention)
    elif opt.distill == 'srrl':
        s_n = feat_s[-1].shape[1]
        t_n = feat_t[-1].shape[1]
        model_fmsr = SRRL(s_n=s_n, t_n=t_n)
        criterion_kd = nn.MSELoss()
        module_list.append(model_fmsr)
        trainable_list.append(model_fmsr)
    elif opt.distill == 'simkd':
        s_n = feat_s[-2].shape[1]
        t_n = feat_t[-2].shape[1]
        model_simkd = SimKD(s_n=s_n, t_n=t_n, factor=opt.factor)
        criterion_kd = nn.MSELoss()
        module_list.append(model_simkd)
        trainable_list.append(model_simkd)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    module_list.append(model_t)

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                # for module in module_list:
                DDP = torch.nn.parallel.DistributedDataParallel
                print([opt.gpu])

                distributed_modules.append(DDP(model_s, device_ids=[opt.gpu]))
                # distributed_modules.append(module_list[1].cuda())
                distributed_modules.append(model_t.cuda())
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    print('opt.batch_size', opt.batch_size)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data, _, train_sampler = get_dataloader_sample(dataset=opt.dataset,
                                                                                           batch_size=opt.batch_size,
                                                                                           num_workers=opt.num_workers,
                                                                                           is_sample=True,
                                                                                           k=opt.nce_k,
                                                                                           multiprocessing_distributed=opt.multiprocessing_distributed)
            else:
                train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset,
                                                                                  batch_size=opt.batch_size,
                                                                                  num_workers=opt.num_workers,
                                                                                  multiprocessing_distributed=opt.multiprocessing_distributed)
        else:
            train_loader, val_loader = get_dali_data_loader(opt)

    else:
        if opt.distill in ['crd']:
            train_loader, val_loader, test_loader, train_sampler = get_histo_dataloader_sample(
                opt=opt,
                dataset=opt.dataset,
                batch_size=opt.batch_size, num_workers=opt.num_workers,
                multiprocessing_distributed=opt.multiprocessing_distributed)
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
        teacher_acc, avg, output_stat = validate_vanilla(test_loader, model_t, criterion_cls, opt)
        print(output_stat)

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')

    # routine
    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)
            # No test_sampler because epoch is random seed, not needed in sequential testing.

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        if opt.distill in ['moma']:
            train_acc, train_loss = train_distill_moma(epoch, train_loader, module_list, criterion_list,
                                                          trainer, contrast, optimizer, opt)
        else:
            train_acc, train_loss = train_distill_moma(epoch, train_loader, module_list, criterion_list,
                                                          None, None, optimizer, opt)

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

        val_acc, val_loss, val_output_stat = validate_distill(val_loader, module_list, criterion_cls, opt, prefix='Val')
        for i in val_output_stat.keys():
            print(i, [val_output_stat[i]])

        if not opt.skip_test:
            print('GPU %d testing' % (opt.gpu))
            test_acc, test_loss, test_output_stat = validate_distill(test_loader, module_list, criterion_cls, opt,
                                                                     prefix='Test')
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
            return f / opt.n_cls

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc_val@1 {:.3f}'.format(val_acc))
            print(' ** Best Acc_val@1 {:.3f}'.format(best_acc))
            val_f1 = f1(val_output_stat['conf_mat'])

            if not opt.skip_test:
                print(' ** Acc_test@1 {:.3f}'.format(test_acc))

            logger.log_value('val_acc', val_acc, epoch)
            logger.log_value('val_loss', val_loss, epoch)

            if not opt.skip_test:
                logger.log_value('test_acc', test_acc, epoch)
                logger.log_value('test_loss', test_loss, epoch)

            # Save all
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }

            print(opt.save_folder)
            # save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                state['best_acc'] = best_acc
                state['best_acc_epoch'] = epoch
                save_file = os.path.join(opt.save_folder, 'net_best_acc.pth')
                print('saving the best acc model!')
                torch.save(state, save_file)

            # save the best f1 model
            if val_f1 > best_f1:
                best_f1 = val_f1
                state['best_f1'] = best_f1
                state['best_f1_epoch'] = epoch
                save_file = os.path.join(opt.save_folder, 'net_best_f1.pth')
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
        save_state = {k: v for k, v in opt._get_kwargs()}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model_s.parameters()) / 1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] = (time.time() - total_time) / 3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(save_state, params_json_path)


if __name__ == '__main__':
    main()
