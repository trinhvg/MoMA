import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from math import pi


eps = 1e-7


class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    @staticmethod
    def forward(x):
        return torch.flatten(x, 1)


# Fourier feature mapping
def input_mapping_torch(x, B_w, B_b):
    x_proj = torch.matmul(x, B_w) + B_b
    return torch.cos(x_proj)

class RFF_ST(nn.Module):
    def __init__(self, w_scale=1., b_scale=1., b_init='gauss01', RFF_init='gauss01', out_dim=128, ):
        super(RFF_ST, self).__init__()
        self.w_scale = w_scale
        self.b_scale = b_scale
        self.b_init = b_init
        self.out_dim = out_dim
        self.RFF_init = RFF_init

    def forward(self, x, xt):
        x = x.flatten(start_dim=1)
        xt = xt.flatten(start_dim=1)

        """"self.RFF_init == 'gauss01'"""
        B_w = torch.empty((x.shape[-1], self.out_dim)).normal_(mean=0, std=1).cuda()

        """ b_init = uniform """
        B_b = torch.distributions.uniform.Uniform(0, 6.283).sample([1, self.out_dim]).cuda()

        B_w *= self.w_scale
        B_b *= self.b_scale

        out = input_mapping_torch(x, B_w, B_b)
        out_t = input_mapping_torch(xt, B_w, B_b)
        return out, out_t


class RFF(nn.Module):
    def __init__(self, w_scale=1., b_scale=1., b_init='gauss01', RFF_init='gauss01', out_dim=128, ):
        super(RFF, self).__init__()
        self.w_scale = w_scale
        self.b_scale = b_scale
        self.b_init = b_init
        self.out_dim = out_dim
        self.RFF_init = RFF_init

    def forward(self, x):
        x = x.flatten(start_dim=1)

        """"self.RFF_init == 'gauss01'"""
        B_w = torch.empty((x.shape[-1], self.out_dim)).normal_(mean=0, std=1).cuda()

        """ b_init = uniform """
        B_b = torch.distributions.uniform.Uniform(0, 6.283).sample([1, self.out_dim]).cuda()

        B_w *= self.w_scale
        B_b *= self.b_scale

        out = input_mapping_torch(x, B_w, B_b)
        out = (2 / self.in_dim) ** 0.5 * out

        return out


class RFF_fixed(nn.Module):
    def __init__(self, in_dim, w_scale=1., b_scale=1., b_init='gauss01', RFF_init='gauss01', out_dim=128, ):
        super(RFF_fixed, self).__init__()
        self.w_scale = w_scale
        self.b_scale = b_scale
        self.b_init = b_init
        self.out_dim = out_dim
        self.RFF_init = RFF_init
        self.in_dim = in_dim

        """"self.RFF_init == 'gauss01'"""
        self.B_w = torch.empty((in_dim, self.out_dim)).normal_(mean=0, std=1).cuda()

        """ b_init = uniform """
        self.B_b = torch.distributions.uniform.Uniform(0, 6.283).sample([1, self.out_dim]).cuda()

        self.B_w *= self.w_scale
        self.B_b *= self.b_scale

    def forward(self, x):
        x = x.flatten(start_dim=1)
        out = input_mapping_torch(x, self.B_w, self.B_b)
        out = (2 / self.in_dim) ** 0.5 * out
        return out


# class Attention_(nn.Module):
#     def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #dim = 384, dim = 64
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         N, C = x.shape # B, N, C = x.shape #torch.Size([4, 197, 384])
#         qkv = self.qkv(x).reshape(N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #torch.Size([4, 197, 1152]) => torch.Size([4, 197, 3, 12, 32]) => torch.Size([3, 4, 12, 197, 32]) # 3 because we need 3 seperate vectors qkv
#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # 3 x torch.Size([4, 12, 197, 32])
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale #torch.Size([4, 12, 197, 197])   torch.Size([4, 12, 197, 32])@ torch.Size([4, 12, 32, 197])
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(N, C) #torch.Size([4, 12, 197, 32]) => torch.Size([4, 197, 12, 32]) => torch.Size([4, 197, 384])
#         x = self.proj(x) # torch.Size([4, 197, 384])
#         x = self.proj_drop(x) #torch.Size([4, 197, 384
#         return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #dim = 384, dim = 64
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # 196 = 14**2, 14 = 224/16
        x = x.unsqueeze(0)
        B, N, C = x.shape #torch.Size([4, 197, 384])
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #torch.Size([4, 197, 1152]) => torch.Size([4, 197, 3, 12, 32]) => torch.Size([3, 4, 12, 197, 32])
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # 3 x torch.Size([4, 12, 197, 32])

        attn = (q @ k.transpose(-2, -1)) * self.scale #torch.Size([4, 12, 197, 197])   torch.Size([4, 12, 197, 32])@ torch.Size([4, 12, 32, 197])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, C) #torch.Size([4, 12, 197, 32]) => torch.Size([4, 197, 12, 32]) => torch.Size([4, 197, 384])
        x = self.proj(x) # torch.Size([4, 197, 384])
        x = self.proj_drop(x) #torch.Size([4, 197, 384
        return x



class Attention_viz(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #dim = 384, dim = 64
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # 196 = 14**2, 14 = 224/16
        x = x.unsqueeze(0)
        B, N, C = x.shape #torch.Size([4, 197, 384])
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #torch.Size([4, 197, 1152]) => torch.Size([4, 197, 3, 12, 32]) => torch.Size([3, 4, 12, 197, 32])
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # 3 x torch.Size([4, 12, 197, 32])

        attn = (q @ k.transpose(-2, -1)) * self.scale #torch.Size([4, 12, 197, 197])  = torch.Size([4, 12, 197, 32])@ torch.Size([4, 12, 32, 197])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, C) #torch.Size([4, 12, 197, 32]) => torch.Size([4, 197, 12, 32]) => torch.Size([4, 197, 384])
        x = self.proj(x) # torch.Size([4, 197, 384])
        x = self.proj_drop(x) #torch.Size([4, 197, 384
        return x, attn



class Attention_(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #dim = 384, dim = 64
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, C = x.shape # B, N, C = x.shape #torch.Size([4, 197, 384])
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #torch.Size([4, 197, 1152]) => torch.Size([4, 197, 3, 12, 32]) => torch.Size([3, 4, 12, 197, 32]) # 3 because we need 3 seperate vectors qkv
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # 3 x torch.Size([4, 12, 197, 32])

        attn = (q @ k.transpose(-2, -1)) * self.scale #torch.Size([4, 12, 197, 197])   torch.Size([4, 12, 197, 32])@ torch.Size([4, 12, 32, 197])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, C) #torch.Size([4, 12, 197, 32]) => torch.Size([4, 197, 12, 32]) => torch.Size([4, 197, 384])
        x = self.proj(x) # torch.Size([4, 197, 384])
        x = self.proj_drop(x) #torch.Size([4, 197, 384
        return x

class Attention2(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.attn_layer = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(self.attn_layer(x)+x)


class CMO(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """

    def __init__(self, opt):
        super(CMO, self).__init__()
        self.opt = opt
        if opt.head == 'mlp':
            self.embed_s = nn.Sequential(
                Flatten(),
                nn.Linear(opt.s_dim, opt.s_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.s_dim, opt.feat_dim),
                Normalize(2)
            )
            self.embed_t = nn.Sequential(
                Flatten(),
                nn.Linear(opt.t_dim, opt.t_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.t_dim, opt.feat_dim),
                Normalize(2)
            )
        elif opt.head == 'mlp_byol':
            self.embed_s = nn.Sequential(
                Flatten(),
                nn.Linear(opt.s_dim, opt.s_dim),
                nn.BatchNorm1d(opt.s_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.s_dim, opt.feat_dim),
                Normalize(2)
            )
            self.embed_t = nn.Sequential(
                Flatten(),
                nn.Linear(opt.t_dim, opt.t_dim),
                nn.BatchNorm1d(opt.t_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.t_dim, opt.feat_dim),
                Normalize(2)
            )
        elif opt.head == 'linear':
            self.embed_s = nn.Sequential(
                Flatten(),
                nn.Linear(opt.s_dim, opt.feat_dim),
                Normalize(2)
            )
            self.embed_t = nn.Sequential(
                Flatten(),
                nn.Linear(opt.t_dim, opt.feat_dim),
                Normalize(2)
            )
        else:
            self.embed_s = nn.Sequential(
                Flatten(),
                Normalize(2)
            )
            self.embed_t = nn.Sequential(
                Flatten(),
                Normalize(2)
            )

        self.norm1 = nn.LayerNorm
        self.qkv_bias = True
        if opt.attn in ['all', 'self_mix']:
            self.atts = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
        elif opt.attn == 'qk':
            self.atts = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
        elif opt.attn in ['dual', 'dual2']:
            self.atts_p = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_n = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
        elif opt.attn in ['self_qk', 'self_nomix']:
            # opt.attn == 'self'
            self.atts_q = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_k = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
        elif opt.attn in ['self_qkv2']:
            # opt.attn == 'self'
            self.atts_q = Attention2(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_k = Attention2(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
        elif opt.attn in ['selfv2', ]:
            # opt.attn == 'self'
            self.atts_q = Attention2(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_k = Attention2(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_queue = Attention2(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
        elif opt.attn == 'self_viz':
            # opt.attn == 'self'
            self.atts_q = Attention_viz(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_k = Attention_viz(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_queue = Attention_viz(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
        else:
            # opt.attn == 'self'
            self.atts_q = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_k = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)
            self.atts_queue = Attention(opt.feat_dim, num_heads=4, qkv_bias= self.qkv_bias, attn_drop=0., proj_drop=0.)





class CMO_EmaTec(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """

    def __init__(self, opt):
        super(CMO_EmaTec, self).__init__()
        self.opt = opt
        if opt.head == 'mlp':
            self.embed_s = nn.Sequential(
                Flatten(),
                nn.Linear(opt.s_dim, opt.s_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.t_dim, opt.feat_dim),
                Normalize(2)
            )
            self.embed_ema = nn.Sequential(
                Flatten(),
                nn.Linear(opt.s_dim, opt.s_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.t_dim, opt.feat_dim),
                Normalize(2)
            )
            self.embed_t = nn.Sequential(
                Flatten(),
                nn.Linear(opt.t_dim, opt.t_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.t_dim, opt.feat_dim),
                Normalize(2)
            )
        elif opt.head == 'RFF_fixed':
            self.embed_s = RFF_fixed(in_dim=opt.s_dim, out_dim=opt.feat_dim)
            self.embed_ema = RFF_fixed(in_dim=opt.s_dim, out_dim=opt.feat_dim)
            self.embed_t = RFF_fixed(in_dim=opt.s_dim, out_dim=opt.feat_dim)
        elif opt.head == 'RFF':
            self.embed_s = RFF_fixed(in_dim=opt.s_dim, out_dim=opt.feat_dim)
            self.embed_ema = RFF_fixed(in_dim=opt.s_dim, out_dim=opt.feat_dim)
            self.embed_t = RFF_fixed(in_dim=opt.s_dim, out_dim=opt.feat_dim)
        else:
            self.embed_s = nn.Sequential(
                Flatten(),
                Normalize(2)
            )
            self.embed_ema = nn.Sequential(
                Flatten(),
                Normalize(2)
            )
            self.embed_t = nn.Sequential(
                Flatten(),
                Normalize(2)
            )

    def forward(self, f_s, f_ema, f_t):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_ema = self.embed_ema(f_ema)
        f_t = self.embed_t(f_t)
        return f_s, f_ema, f_t