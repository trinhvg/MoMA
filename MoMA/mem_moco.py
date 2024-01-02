import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMoCo(nn.Module):
    """base class for MoCo-style memory cache"""
    def __init__(self, K=65536, T=0.07):
        super(BaseMoCo, self).__init__()
        self.K = K
        self.T = T
        self.index = 0

    def _update_pointer(self, bsz):
        self.index = (self.index + bsz) % self.K

    def _update_memory(self, k, queue):
        """
        Args:
          k: key feature
          queue: memory buffer
        """
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index, self.K).long()
            queue.index_copy_(0, out_ids, k)

    def _compute_logit(self, q, k, queue):
        """
        Args:
          q: query/anchor feature
          k: key feature
          queue: memory buffer
        """
        # pos logit
        bsz = q.shape[0]
        pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1)) # torch.Size([64, 1, 1280]) bmm torch.Size([64, 1280, 1]) = torch.Size([64, 1, 1])
        pos = pos.view(bsz, 1) #torch.Size([64, 1])

        # neg logit
        neg = torch.mm(queue, q.transpose(1, 0)) # torch.Size([16384, 512]) bmm  bmm torch.Size([512*64]) = torch.Size([16384, 64])
        neg = neg.transpose(0, 1) #torch.Size([64, 16384])

        out = torch.cat((pos, neg), dim=1) #torch.Size([64, 16385])
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        return out

    def _compute_logit_qk(self, q, k):
        """
        Args:
          q: query/anchor feature
          k: key feature
          queue: memory buffer
        """
        # pos logit
        bsz = q.shape[0]
        pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1)) # torch.Size([64, 1, 1280]) bmm torch.Size([64, 1280, 1]) = torch.Size([64, 1, 1])
        pos = pos.view(bsz, 1) #torch.Size([64, 1])

        out = torch.div(pos, self.T)
        out = out.squeeze().contiguous()

        return out


class MoCo(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""
    def __init__(self, n_dim, K=65536, T=0.07, mem_name='memory'):
        super(MoCo, self).__init__(K, T)
        # create memory queue
        self.register_buffer(mem_name, torch.randn(K, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, q, k, all_k=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
        """
        bsz = q.size(0)
        k = k.detach()

        # compute logit
        queue = self.memory.clone().detach()
        logits = self._compute_logit(q, k, queue)


        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k = all_k if all_k is not None else k
        self._update_memory(all_k, self.memory)
        self._update_pointer(all_k.size(0))
        return logits, labels


class MoCoAtt(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""
    def __init__(self, n_dim, K=65536, T=0.07, mem_name='memory'):
        super(MoCoAtt, self).__init__(K, T)
        # create memory queue
        self.register_buffer(mem_name, torch.randn(K, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, q, k, all_k=None, attn=None, criterion_kd=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
        """
        bsz = q.size(0)
        k = k.detach()

        # compute logit
        queue = self.memory.clone().detach()
        if attn == 'all':
            attn_out = criterion_kd.atts(torch.concat([q, k, queue], dim=0))
            q, k, queue = attn_out[:bsz], attn_out[bsz:2 * bsz], attn_out[2 * bsz:]
        elif attn == 'qk':
            attn_out = criterion_kd.atts(torch.concat([q, k], dim=0))
            q, k = attn_out[:bsz], attn_out[bsz:]
        elif attn == 'dual':
            attn_out_p = criterion_kd.atts_p(torch.concat([q, queue], dim=0))
            q, queue = attn_out_p[:bsz], attn_out_p[bsz:]
            attn_out_n = criterion_kd.atts_n(torch.concat([k, queue], dim=0))
            k, queue = attn_out_n[:bsz], attn_out_n[bsz:]
        elif attn == 'dual2':
            attn_out_p = criterion_kd.atts_p(torch.concat([q, k], dim=0))
            q = attn_out_p[:bsz]
            attn_out_n = criterion_kd.atts_n(torch.concat([k, q], dim=0))
            k = attn_out_n[:bsz]
        elif attn in ['self_qk', 'self_qkv2']:
            q = criterion_kd.atts_q(q)
            k = criterion_kd.atts_k(k)
        else:
            q = criterion_kd.atts_q(q)
            k = criterion_kd.atts_k(k)
            queue = criterion_kd.atts_queue(queue)

        if attn == 'dual2':
            logits = self._compute_logit_qk(q, k)
        else:
            logits = self._compute_logit(q, k, queue)


        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k = all_k if all_k is not None else k
        self._update_memory(all_k, self.memory)
        self._update_pointer(all_k.size(0))
        return logits, labels



class MoCoST(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""
    def __init__(self, n_dim, K=65536, T=0.07):
        super(MoCoST, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory_s', torch.randn(K, n_dim))
        self.register_buffer('memory_t', torch.randn(K, n_dim))
        self.memory_s = F.normalize(self.memory_s)
        self.memory_t = F.normalize(self.memory_t)
        # we can mix s and t into to 1 memory

    def forward(self, q, k, k_t, all_k=None, all_k_t=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
        """
        bsz = q.size(0)
        k = k.detach()
        k_t = k_t.detach()

        # compute logit
        queue_s = self.memory_s.clone().detach()
        queue_t = self.memory_t.clone().detach()
        logits_ss = self._compute_logit(q, k, queue_s)
        logits_st = self._compute_logit(q, k_t, queue_t)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k = all_k if all_k is not None else k
        all_k_t = all_k_t if all_k_t is not None else k_t
        self._update_memory(all_k, self.memory_s)
        self._update_memory(all_k_t, self.memory_t)
        self._update_pointer(all_k.size(0))

        return logits_ss, logits_st, labels



class MoCoSSTT(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""
    def __init__(self, n_dim, K=65536, T=0.07):
        super(MoCoSSTT, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory_s', torch.randn(K, n_dim))
        self.register_buffer('memory_t', torch.randn(K, n_dim))
        self.memory_s = F.normalize(self.memory_s)
        self.memory_t = F.normalize(self.memory_t)
        # we can mix s and t on to 1 memory

    def forward(self, q, k, q_t=None, k_t=None, all_k=None, all_k_t=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
        """
        bsz = q.size(0)
        k = k.detach()
        k_t = k_t.detach()

        # compute logit
        queue_s = self.memory_s.clone().detach()
        queue_t = self.memory_t.clone().detach()
        logits_ss = self._compute_logit(q, k, queue_s)
        logits_st = self._compute_logit(q, k_t, queue_t)
        if q_t is not None:
            logits_ts = self._compute_logit(q_t, k, queue_s)
            logits_tt = self._compute_logit(q_t, k_t, queue_t)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k = all_k if all_k is not None else k
        all_k_t = all_k_t if all_k_t is not None else k_t
        self._update_memory(all_k, self.memory_s)
        self._update_memory(all_k_t, self.memory_t)
        self._update_pointer(all_k.size(0))

        if q_t is not None:
            return logits_ss, logits_st, logits_ts, logits_tt, labels
        else:
            return logits_ss, logits_st, labels


def build_mem(opt):
    if opt.mem == 'MoCoSSTT':
        mem_func = MoCoSSTT
        memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)

    elif opt.mem == 'MoCoST':
        mem_func = MoCoST
        memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)

    elif opt.mem == 'MoCoAtt':
        mem_func = MoCoAtt
        memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)

    else:
        mem_func = MoCo
        memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)

    return memory