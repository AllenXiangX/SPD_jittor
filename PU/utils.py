import sys
sys.path.append('..')
import yaml
import jittor as jt
from jittor.contrib import concat
from models.utils import FurthestPointSampler, grouping_operation, knn
from easydict import EasyDict as edict
from loss_functions import chamfer_unidirectional_l2


def create_edict(pack):
    d = edict()
    for key, value in pack.items():
        if isinstance(value, dict):
            d[key] = create_edict(value)
        else:
            d[key] = value
    return d


def read_yaml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return create_edict(config)




def knn_sample(p1, p2, k=256):
    """
    Args:
        p1: b, s, 3
        p2: b, n, 3
    """
    _, idx_knn = knn(p1, p2, k)

    point_groups = grouping_operation(p2.permute(0, 2, 1), idx_knn).permute(0, 2, 3, 1)
    return point_groups

def patch_extraction(point_clouds, num_per_patch=256, patch_num_ratio=3):
    """
    Args:
        point_clouds: b, n, 3
    """
    b, n, _ = point_clouds.shape
    seed_num = int(n / num_per_patch * patch_num_ratio)

    seed_points = FurthestPointSampler(seed_num)(point_clouds)

    patch_points = knn_sample(seed_points, point_clouds, k=num_per_patch)
    patch_points = patch_points.reshape((b*seed_num, num_per_patch, 3))

    return patch_points

def random_subsample(pcd, n_points=256):
    """
    Args:
        pcd: (B, N, 3)

    returns:
        new_pcd: (B, n_points, 3)
    """
    b, n, _ = pcd.shape
    batch_idx = jt.arange(b, dtype=jt.long).reshape((-1, 1)).repeat(1, n_points)
    idx = concat([jt.randperm(n, dtype=jt.long)[:n_points].reshape((1, -1)) for i in range(b)], 0)
    return pcd[batch_idx, idx, :]


def chamfer_radius(p1, p2, radius=1.0):
    d1 = chamfer_unidirectional_l2(p1, p2)
    d2 = chamfer_unidirectional_l2(p2, p1)
    cd_dist = 0.5 * d1 + 0.5 * d2
    cd_dist = jt.mean(cd_dist, dim=1)
    cd_dist_norm = cd_dist / radius
    cd_loss = jt.mean(cd_dist_norm)
    return cd_loss

class PULoss:
    def __init__(self):
        self.sampler_256 = FurthestPointSampler(256)

    def get_loss(self, pcds, gt, radius):
        """
            Args:
                pcds: list of point clouds, [256, 512, 1048, 1048]
            """
        p1, p2, p3, p4 = pcds
        gt_1 = self.sampler_256(gt)
        cd_1 = chamfer_radius(p1, gt_1, radius)

        cd_3 = chamfer_radius(p3, gt, radius)
        cd_4 = chamfer_radius(p4, gt, radius)

        return cd_1 + cd_3 + cd_4, cd_4


class LambdaLR(object):
    def __init__(self, optimizer, base_lr, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lr = base_lr
        self.last_epoch = last_epoch
        # TODO set last_epoch is not ready

    def get_gamma(self):
        return self.lr_lambda(self.last_epoch)

    def get_lr(self):
        now_lr = self.optimizer.lr
        return now_lr * self.get_gamma()

    def step(self):
        self.last_epoch += 1
        self.update_lr()

    def update_lr(self):
        gamma = self.get_gamma()
        if gamma != 1.0:
            self.optimizer.lr = self.optimizer.lr * gamma
            for i, param_group in enumerate(self.optimizer.param_groups):
                if param_group.get("lr") != None:
                    param_group["lr"] = self.base_lr * gamma


