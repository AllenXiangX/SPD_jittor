# -*- coding: utf-8 -*-
# @Author: Peng Xiang
import numpy as np
import jittor as jt
from jittor import nn, init
from jittor.contrib import concat
from models.misc.ops import knn, gather_operation, grouping_operation
from models.misc.ops import FurthestPointSampler


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def execute(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(nn.relu(self.conv_1(x))) + shortcut
        return out

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def execute(self, inputs):
        return self.mlp(inputs)


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def execute(self, x, pos):
        """
        Args:
            x: Tensor, (B, c, 2048)
            pos: Tensor, (B, 2048, 3)
        """
        identity = x
        x_bcn = self.linear_start(x)
        b, dim, n = x_bcn.shape
        pos_bcn = pos.transpose(0, 2, 1)
        _, idx_knn = knn(pos, pos, self.n_knn)
        # idx_knn = knn(pos_bcn, self.n_knn)

        key = self.conv_key(x_bcn)
        value = self.conv_value(x_bcn)
        query = self.conv_query(x_bcn)

        # key = index_points(key.transpose(0, 2, 1), idx_knn).transpose(0, 3, 1, 2)  # (b, c, n, n_knn)
        key = grouping_operation(key, idx_knn)
        # print('key.shape', key.shape)
        qk_rel = query.reshape((b, -1, n, 1)) - key


        pos_rel = pos_bcn.reshape((b, -1, n, 1)) - \
                  grouping_operation(pos_bcn, idx_knn)
                  # index_points(pos, idx_knn).transpose(0, 3, 1, 2)
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = nn.softmax(attention, dim=-1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = (value * attention).sum(dim=-1)
        y = self.linear_end(agg)

        return y+identity


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=nn.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def execute(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, nsample, 3)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, nsample, _ = xyz.shape
    new_xyz = jt.zeros((1, 1, 3), dtype=jt.float).repeat(b, 1, 1)
    xyz_b3n = xyz.permute(0, 2, 1)
    grouped_xyz = xyz_b3n.reshape((b, 3, 1, nsample))
    idx = jt.arange(nsample).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = concat([xyz_b3n, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)
        self.sampler = FurthestPointSampler(npoint)

    def execute(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, N, 3)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(xyz, points, self.sampler, self.nsample, self.use_xyz, idx=idx)

        new_points = self.mlp_conv(new_points)
        new_points = jt.max(new_points, 3)

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


def sample_and_group_knn(xyz, points, sampler, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, N, 3)
        points: Tensor, (B, f, N)
        sampler: fps sampler, to sample npoints
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    # new_xyz =
    # new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)
    new_xyz = sampler(xyz) # B, npoint, 3
    if idx is None:
        # idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
        _, idx = knn(new_xyz, xyz, k)

    grouped_xyz = grouping_operation(xyz.permute(0, 2, 1), idx) # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.permute(0, 2, 1).unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = concat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

class CouplingLayer(nn.Module):

    def __init__(self, d, intermediate_dim, swap=False):
        super(CouplingLayer, self).__init__()
        self.d = d - (d // 2)
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

    def execute(self, x, logpx=None, reverse=False):

        if self.swap:
            x = concat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = jt.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        logdetjac = jt.sum(jt.log(scale).view(scale.shape[0], -1), 1, keepdims=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        y = concat([x[:, :self.d], y1], 1) if not self.swap else concat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp

class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def execute(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx

def build_latent_flow(args):
    chain = []
    for i in range(args.latent_flow_depth):
        chain.append(CouplingLayer(args.latent_dim, args.latent_flow_hidden_dim, swap=(i % 2 == 0)))
    return SequentialFlow(chain)



def reparameterize_gaussian(mean, logvar):
    std = jt.exp(0.5 * logvar)
    eps = jt.randn(std.size()).to(mean)
    return mean + std * eps

def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + jt.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdims=False) + const
    return ent

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2

def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.argmax(-1, keepdims=True)[0]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

