import sys
sys.path.append('..')
import random
import jittor as jt
import jittor.nn as nn
from jittor.contrib import concat

def build_lambda_sche(opti, config, last_epoch=-1):
    raise NotImplementedError()
    return scheduler


def seprate_point_cloud(xyz, num_points, crop, fps_sampler, inp_n_points=2048, fixed_points=None, padding_zeros=False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = jt.normalize(jt.randn(1, 1, 3), p=2, dim=-1)
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3)

        distance_matrix = jt.norm(center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1)  # 1 1 2048

        idx = jt.argsort(distance_matrix, dim=-1, descending=False)[0][0][0]  # 2048
        # print('idx.shape', idx.shape)
        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps_sampler(input_data))
            CROP.append(fps_sampler(crop_data))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = concat(INPUT, dim=0)  # B N 3
    crop_data = concat(CROP, dim=0)  # B M 3

    input_data = fps_sampler(input_data)
    return input_data, crop_data
