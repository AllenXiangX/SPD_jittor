# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import jittor as jt
from jittor import nn
from jittor.contrib import concat
from .utils import MLP_Res, grouping_operation, knn


class SkipTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

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

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def execute(self, pos, key, query):
        """
        Args:
            pos: (B, N, 3)
            key: (B, in_channel, N)
            query: (B, in_channel, N)

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """

        value = self.mlp_v(concat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        # pos_flipped = pos.permute(0, 2, 1).contiguous()
        _, idx_knn = knn(pos, pos, self.n_knn)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_b3n = pos.permute(0, 2, 1) # b, 3, n
        pos_rel = pos_b3n.reshape((b, -1, n, 1)) - grouping_operation(pos_b3n, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = nn.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding  #

        # agg = jt.linalg.einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        agg = (value * attention).sum(dim=-1)
        y = self.conv_end(agg)

        return y + identity