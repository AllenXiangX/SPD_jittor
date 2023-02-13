import sys
sys.path.append('..')

from loss_functions import chamfer_l2 as chamfer
from models.utils import FurthestPointSampler

class AELoss:
    def __init__(self):
        self.fps_sampler_512 = FurthestPointSampler(512)

    def get_loss(self, pcds, gt):
        x_512 = self.fps_sampler_512(gt)
        cd_1 = chamfer(pcds[0], x_512)
        cd_3 = chamfer(pcds[-1], gt)
        return cd_1 + cd_3
