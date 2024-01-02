from __future__ import print_function

import os
import sys
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .util import AverageMeter, accuracy
from .base_trainer import BaseTrainer

try:
    from apex import amp, optimizers
except ImportError:
    pass


class ContrastTrainer(BaseTrainer):
    """trainer for contrastive pretraining"""
    def __init__(self, args):
        super(ContrastTrainer, self).__init__(args)

    def logging(self, epoch, logs, lr):
        """ logging to tensorboard

        Args:
          epoch: training epoch
          logs: loss and accuracy
          lr: learning rate
        """
        args = self.args
        if args.rank == 0:
            self.logger.log_value('loss', logs[0], epoch)
            self.logger.log_value('acc', logs[1], epoch)
            self.logger.log_value('jig_loss', logs[2], epoch)
            self.logger.log_value('jig_acc', logs[3], epoch)
            self.logger.log_value('learning_rate', lr, epoch)

    def wrap_up(self, model, model_ema, optimizer):
        """Wrap up models with apex and DDP

        Args:
          model: model
          model_ema: momentum encoder
          optimizer: optimizer
        """
        args = self.args

        model.cuda(args.gpu)
        if isinstance(model_ema, torch.nn.Module):
            model_ema.cuda(args.gpu)

        # to amp model if needed
        if args.amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.opt_level
            )
            if isinstance(model_ema, torch.nn.Module):
                model_ema = amp.initialize(
                    model_ema, opt_level=args.opt_level
                )
        # to distributed data parallel
        model = DDP(model, device_ids=[args.gpu])

        if isinstance(model_ema, torch.nn.Module):
            self.momentum_update(model.module, model_ema, 0)

        return model, model_ema, optimizer

    def broadcast_memory(self, contrast):
        """Synchronize memory buffers

        Args:
          contrast: memory.
        """
        if self.args.mem in ['MoCo', 'MoCoAtt']:
            dist.broadcast(contrast.memory, 0)
        else:
            dist.broadcast(contrast.memory_s, 0)
            dist.broadcast(contrast.memory_t, 0)

    @staticmethod
    def _global_gather(x):
        all_x = [torch.ones_like(x)
                 for _ in range(dist.get_world_size())]
        dist.all_gather(all_x, x, async_op=False)
        return torch.cat(all_x, dim=0)

    def _shuffle_bn(self, x, model_ema, model_ema_head):
        """ Shuffle BN implementation

        Args:
          x: input image on each GPU/process
          model_ema: momentum encoder on each GPU/process
        """
        args = self.args
        local_gp = self.local_group
        bsz = x.size(0)

        # gather x locally for each node [torch.Size([64, 3, 512, 512]), torch.Size([64, 3, 512, 512])]
        node_x = [torch.ones_like(x)
                  for _ in range(dist.get_world_size(local_gp))]
        dist.all_gather(node_x, x.contiguous(),
                        group=local_gp, async_op=False)
        node_x = torch.cat(node_x, dim=0)

        # shuffle bn
        shuffle_ids = torch.randperm(
            bsz * dist.get_world_size(local_gp)).cuda()
        reverse_ids = torch.argsort(shuffle_ids)
        dist.broadcast(shuffle_ids, 0)
        dist.broadcast(reverse_ids, 0)

        this_ids = shuffle_ids[args.local_rank*bsz:(args.local_rank+1)*bsz]
        with torch.no_grad():
            this_x = node_x[this_ids]
            # k = model_ema(this_x, is_feat=True)
            feat_t, logit_t = model_ema(this_x, is_feat=True)
            feat_t = feat_t[-1]
            k = model_ema_head(feat_t)

        # globally gather k
        all_k = self._global_gather(k)

        # unshuffle bn
        node_id = args.node_rank
        ngpus = args.ngpus_per_node
        node_k = all_k[node_id*ngpus*bsz:(node_id+1)*ngpus*bsz]
        this_ids = reverse_ids[args.local_rank*bsz:(args.local_rank+1)*bsz]
        k = node_k[this_ids]

        return k, all_k

    def _shuffle_bn_attn(self, x, model_ema, model_ema_head, criterion_kd, q):
        """ Shuffle BN implementation

        Args:
          x: input image on each GPU/process
          model_ema: momentum encoder on each GPU/process
        """
        args = self.args
        local_gp = self.local_group
        bsz = x.size(0)

        # gather x locally for each node
        node_x = [torch.ones_like(x)
                  for _ in range(dist.get_world_size(local_gp))]
        dist.all_gather(node_x, x.contiguous(),
                        group=local_gp, async_op=False)
        node_x = torch.cat(node_x, dim=0)

        # shuffle bn
        shuffle_ids = torch.randperm(
            bsz * dist.get_world_size(local_gp)).cuda()
        reverse_ids = torch.argsort(shuffle_ids)
        dist.broadcast(shuffle_ids, 0)
        dist.broadcast(reverse_ids, 0)

        this_ids = shuffle_ids[args.local_rank * bsz:(args.local_rank + 1) * bsz]
        with torch.no_grad():
            this_x = node_x[this_ids]
            # k = model_ema(this_x, is_feat=True)
            # k = model_ema_head(k)
            feat_t, logit_t = model_ema(this_x, is_feat=True)
            feat_t = feat_t[-1]
            k = model_ema_head(feat_t)

        if args.attn == 'self_mix':
            attn_out = criterion_kd.atts(torch.cat([q, k], dim=0))
            q, k = attn_out[:bsz],  attn_out[bsz:]
        else:
            #         if args.attn == 'self_nomix':
            q = criterion_kd.atts_q(q)
            k = criterion_kd.atts_k(k)

        # globally gather k
        all_k = self._global_gather(k)

        # unshuffle bn
        node_id = args.node_rank
        ngpus = args.ngpus_per_node
        node_k = all_k[node_id * ngpus * bsz:(node_id + 1) * ngpus * bsz]
        this_ids = reverse_ids[args.local_rank * bsz:(args.local_rank + 1) * bsz]
        k = node_k[this_ids]

        return q, k, all_k

    @staticmethod
    def _compute_loss_accuracy(logits, target, criterion):
        """
        Args:
          logits: a list of logits, each with a contrastive task
          target: contrastive learning target
          criterion: typically nn.CrossEntropyLoss
        """
        losses = [criterion(logit, target) for logit in logits]


        def acc(l, t):
            acc1 = accuracy(l, t)
            return acc1[0]

        accuracies = [acc(logit, target) for logit in logits]
        return losses, accuracies

    @staticmethod
    def momentum_update(model, model_ema, m):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            p2.data.mul_(m).add_(p1.detach().data, alpha=(1 - m))

            # 	add_(Number alpha, Tensor other)
            # Consider using one of the following signatures instead:
            # 	add_(Tensor other, *, Number alpha) (Triggered internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:882.)
            #   p2.data.mul_(m).add_(1 - m, p1.detach().data)
            # p2.data.mul_(m).add_(p1.detach().data, 1 - m)
