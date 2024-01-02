from __future__ import print_function

import json
import torch
import numpy as np
import torch.distributed as dist
import math
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from collections import OrderedDict
import os



def adjust_learning_rate_V0(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

# def adjust_learning_rate(optimizer, epoch, step, len_epoch, old_lr):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     if epoch < 5:
#         lr = old_lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
#     elif 5 <= epoch < 60: return
#     else:
#         factor = epoch // 30
#         factor -= 1
#         lr = old_lr*(0.1**factor)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    lr = opt.learning_rate
    if opt.cosine:
        eta_min = lr * (opt.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / opt.epochs)) / 2

    else:
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        if steps > 0:
            lr = lr * (opt.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def update_dict_to_json(epoch, d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    if not os.path.isfile(json_path):
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: v for k, v in d.items()}
            s = {epoch:d}
            json.dump(s, f, indent=4)

    with open(json_path) as json_file:
        json_data = json.load(json_file)

    d = {k: v for k, v in d.items()}
    current_epoch_dict = {epoch: d}
    json_data.update(current_epoch_dict)

    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    """Loads json file to dict 

    Args:
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def reduce_tensor(tensor, world_size = 1, op='avg'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size > 1:
        rt = torch.true_divide(rt, world_size)
    return rt

def load_pretrained_weights(model, ckpt_path, gpu=None, multiprocessing_distributed=True, strict=True):
    """load pre-trained weights for encoder
-
    Args:
      model: pretrained encoder, should be frozen
    """
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if multiprocessing_distributed else 0)}

    state_dict = torch.load(ckpt_path, map_location=map_location)
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    encoder_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        encoder_state_dict[k] = v
    if not strict:
        encoder_state_dict.pop('classifier_.1.weight')
        encoder_state_dict.pop('classifier_.1.bias')
        model.load_state_dict(encoder_state_dict, strict=False)
    else:
        model.load_state_dict(encoder_state_dict)
    return model

def load_pretrained_weights_teacher(model, ckpt_path, gpu=None, opt=None, key=False):
    """load pre-trained weights for encoder

    Args:
      model: pretrained encoder, should be frozen
    """
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}

    state_dict = torch.load(ckpt_path, map_location=map_location)
    if key:
        state_dict = state_dict['model']
    encoder_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        encoder_state_dict[k] = v
    model.load_state_dict(encoder_state_dict)
    return model


def process_accumulated_output(output, batch_size, nr_classes):
    #
    def uneven_seq_to_np(seq):
        # print(seq)
        # print((len(seq) - 1) )
        # print(len(seq[-1]) )
        item_count = batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        if len(seq) < 2:
            return seq[0]
        for idx in range(0, len(seq) - 1):
            cat_array[idx * batch_size:
                      (idx + 1) * batch_size] = seq[idx]
        cat_array[(idx + 1) * batch_size:] = seq[-1]
        return cat_array

    proc_output = dict()
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    logit = uneven_seq_to_np(output['logit'])

    pred = np.argmax(logit, axis=-1)
    # pred_c = [covert_dict[pred_c[idx]] for idx in range(len(pred_c))]
    acc = np.mean(pred == true)
    print('acc', acc)
    # print(classification_report(true, pred_c, labels=[0, 1, 2, 3]))
    # confusion matrix
    conf_mat = confusion_matrix(true, pred, labels=np.arange(nr_classes))
    proc_output.update(acc=acc, conf_mat=conf_mat,)
    return proc_output
def process_accumulated_output_attack(output, batch_size, nr_classes):
    #
    def uneven_seq_to_np(seq):
        # print(seq)
        # print((len(seq) - 1) )
        # print(len(seq[-1]) )
        item_count = batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        if len(seq) < 2:
            return seq[0]
        for idx in range(0, len(seq) - 1):
            cat_array[idx * batch_size:
                      (idx + 1) * batch_size] = seq[idx]
        cat_array[(idx + 1) * batch_size:] = seq[-1]
        return cat_array

    proc_output = dict()
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    logit = uneven_seq_to_np(output['logit'])
    logit_attack = uneven_seq_to_np(output['logit_a'])

    pred = np.argmax(logit, axis=-1)
    pred_a = np.argmax(logit_attack, axis=-1)
    # pred_c = [covert_dict[pred_c[idx]] for idx in range(len(pred_c))]
    acc = np.mean(pred == true)
    acc_a = np.mean(pred_a == true)
    pred_final = [pred[i] if pred[i] != true[i] else pred_a[i] for i in range(len(pred))]
    acc_final = np.mean(pred_final == true)

    print('acc', acc)
    # print(classification_report(true, pred_c, labels=[0, 1, 2, 3]))
    # confusion matrix
    conf_mat = confusion_matrix(true, pred_a, labels=np.arange(nr_classes))
    proc_output.update(acc=acc, acc_a=acc_a, acc_final=acc_final, conf_mat=conf_mat,)
    return proc_output

if __name__ == '__main__':

    pass
