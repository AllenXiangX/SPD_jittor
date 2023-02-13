import sys
import numpy as np

sys.path.append('..')
from loss_functions import chamfer_l1, chamfer_l2
from models.utils import FurthestPointSampler


def pc_normalize(pc, radius):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m * radius
    return pc


def calc_cd(output, gt):
    cd_p = chamfer_l1(output, gt)
    cd_t = chamfer_l2(output, gt) # (dist1.mean(1) + dist2.mean(1))

    return cd_p, cd_t

class SVRLoss:
    def __init__(self):
        self.fps_sampler_512 = FurthestPointSampler(512)

    def get_loss(self, outputs, gt):
        x_512 = self.fps_sampler_512(gt)
        p1 = outputs[0]
        p3 = outputs[-1]
        cd1 = chamfer_l1(p1, x_512)
        cd3 = chamfer_l1(p3, gt)
        return cd1 + cd3



