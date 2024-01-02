from __future__ import print_function, division
from cProfile import label

import sys
import time
import torch
from .util import AverageMeter, accuracy, reduce_tensor, process_accumulated_output
import torch.nn as nn
import torch.nn.functional as F
# from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(train_loader)

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        images, labels = batch_data

        # if opt.dali is None:
        #     images, labels = batch_data
        # else:
        #     images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()
        
        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        output = model(images)
        loss = criterion(output, labels)
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, labels, topk=(1,))
        top1.update(metrics[0].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   loss=losses, top1=top1))
            sys.stdout.flush()
            
    return top1.avg, losses.avg


# for comparison with other KD
def train_distill_compare(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(train_loader)

    end = time.time()
    for idx, data in enumerate(train_loader):
        # if opt.dali is None:
        if opt.distill in ['crd']:
            images, labels, index, contrast_idx = data
        else:
            images, labels = data
        # else:
        #     images, labels = data[0]['data'], data[0]['label'].squeeze().long()

        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        if opt.distill != 'cmo':
            with torch.no_grad():
                # print(images.shape)

                feat_t, logit_t = model_t(images, is_feat=True)
                feat_t = [f.detach() for f in feat_t]

        # cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
        cls_t = model_t.get_feat_modules()[-1]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd loss
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            # include 1, exclude -1.
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-2]
            g_t = feat_t[1:-2]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)

        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)
        elif opt.distill == 'srrl':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1].squeeze(-1).squeeze(-1), cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1].squeeze(-1).squeeze(-1)) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
        else:
            raise NotImplementedError(opt.distill)



        loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_kd


        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1,))
        top1.update(metrics[0].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses, top1=top1,
                batch_time=batch_time))
            sys.stdout.flush()

    return top1.avg, losses.avg


def train_distill_moma(epoch, train_loader, module_list, criterion_list, trainer, contrast, optimizer, opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(train_loader)

    end = time.time()
    for idx, data in enumerate(train_loader):
        # if opt.dali is None:
        if opt.distill in ['crd']:
            images, labels, index, contrast_idx = data
        else:
            images, labels = data
        # else:
        #     images, labels = data[0]['data'], data[0]['label'].squeeze().long()

        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()




        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        # if opt.distill != 'cmo':
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        # cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
        cls_t = model_t.get_feat_modules()[-1]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd loss
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            # include 1, exclude -1.
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)

        elif opt.distill == 'moma':
            trainer.momentum_update(model_s.module, model_t, opt.alpha)
            if opt.head == 'mlp':
                criterion_kd.embed_t.eval()
                trainer.momentum_update(criterion_kd.embed_s, criterion_kd.embed_t, opt.alpha)

            def set_bn_train(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.train()
            model_t.apply(set_bn_train)
            # shuffle BN for momentum encoder
            k, all_k = trainer._shuffle_bn(images, model_t, model_ema_head=criterion_kd.embed_t)

            criterion = nn.CrossEntropyLoss().cuda()
            f_s = feat_s[-1]
            f_s= criterion_kd.embed_s(f_s)

            if opt.attn == 'self':
                f_s = criterion_kd.atts_q(f_s)
                k = criterion_kd.atts_k(k)
                all_k = criterion_kd.atts_queue(all_k)

            output = contrast(q=f_s, k=k, all_k=all_k)
            c_losses, c_accuracies = trainer._compute_loss_accuracy(
                logits=output[:-1], target=output[-1],
                criterion=criterion)
            loss_kd = c_losses[0]

        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)
        elif opt.distill == 'srrl':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_kd
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, ))
        top1.update(metrics[0].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses, top1=top1,
                batch_time=batch_time))
            sys.stdout.flush()
    return top1.avg, losses.avg


def validate_vanilla(val_loader, model, criterion, opt,  prefix='Test'):
    """validation"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    infer_output = ['logit', 'true']
    accumulator = {metric: [] for metric in infer_output}
    # switch to evaluate mode
    model.eval()

    # n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            images, labels = batch_data
            # if opt.dali is None:
            #     images, labels = batch_data
            # else:
            #     images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1,))
            top1.update(metrics[0].item(), images.size(0))
            batch_time.update(time.time() - end)
            accumulator['logit'].extend([output.cpu().numpy()])
            accumulator['true'].extend([labels.cpu().numpy()])

            if idx % opt.print_freq == 0:
                print('{3}: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, prefix, batch_time=batch_time, loss=losses,
                       top1=top1))
    output_stat = process_accumulated_output(accumulator, opt.batch_size, opt.n_cls)

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, losses.count]).to(opt.gpu)
        cf_metrics = torch.tensor(output_stat['conf_mat']).to(opt.gpu)

        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        cf_metrics = reduce_tensor(cf_metrics, 1)

        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        stat = {'acc': ret[0], 'conf_mat': cf_metrics.cpu().numpy()}

        return ret[0], ret[1], stat

    return top1.avg, losses.avg, output_stat



def validate_distill(val_loader, module_list, criterion, opt, prefix='Test'):
    """validation"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    infer_output = ['logit', 'true']
    accumulator = {metric: [] for metric in infer_output}

    # switch to evaluate mode
    for module in module_list:
        module.eval()
    
    model_s = module_list[0]
    model_t = module_list[-1]
    # n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            images, labels = batch_data
            # if opt.dali is None:
            #     images, labels = batch_data
            # else:
            #     images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            if opt.distill == 'simkd':
                feat_s, _ = model_s(images, is_feat=True)
                feat_t, _ = model_t(images, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                # cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
                cls_t = model_t.get_feat_modules()[-1]
                _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            else:
                output = model_s(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, ))
            top1.update(metrics[0].item(), images.size(0))
            batch_time.update(time.time() - end)
            accumulator['logit'].extend([output.cpu().numpy()])
            accumulator['true'].extend([labels.cpu().numpy()])
            
            if idx % opt.print_freq == 0:
                print('{3}: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, prefix, batch_time=batch_time, loss=losses,
                       top1=top1))
    output_stat = process_accumulated_output(accumulator, opt.batch_size, opt.n_cls)

                
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, losses.count]).to(opt.gpu)
        cf_metrics = torch.tensor(output_stat['conf_mat']).to(opt.gpu)

        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        cf_metrics = reduce_tensor(cf_metrics, 1)

        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))

        stat = {'acc': ret[0], 'conf_mat': cf_metrics.cpu().numpy()}

        return  ret[0], ret[1], stat

    return top1.avg, losses.avg, output_stat






def validate_distill_cmo(val_loader, module_list, criterion, opt, prefix='Test'):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    infer_output = ['logit', 'true']
    accumulator = {metric: [] for metric in infer_output}

    # switch to evaluate mode
    for module in module_list:
        module.eval()

    model_s = module_list[0]
    model_t = module_list[-1]
    # n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            images, labels = batch_data
            # if opt.dali is None:
            #     images, labels = batch_data
            # else:
            #     images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            if opt.distill == 'simkd':
                feat_s, _ = model_s(images, is_feat=True)
                feat_t, _ = model_t(images, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
                model_t.get_feat_modules()[-1]
                _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            else:
                output = model_s(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1,))
            top1.update(metrics[0].item(), images.size(0))
            batch_time.update(time.time() - end)
            accumulator['logit'].extend([output.cpu().numpy()])
            accumulator['true'].extend([labels.cpu().numpy()])

            if idx % opt.print_freq == 0:
                print('{3}: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, prefix, batch_time=batch_time, loss=losses,
                    top1=top1))
    output_stat = process_accumulated_output(accumulator, opt.batch_size, opt.n_cls)

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, losses.count]).to(opt.gpu)
        cf_metrics = torch.tensor(output_stat['conf_mat']).to(opt.gpu)

        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        cf_metrics = reduce_tensor(cf_metrics, 1)

        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))

        stat = {'acc': ret[0], 'conf_mat': cf_metrics.cpu().numpy()}

        return ret[0], ret[1], stat

    return top1.avg, losses.avg, output_stat
