import sys
import jittor as jt
sys.path.append('..')

from loss_functions import chamfer_l1, chamfer_l2
from models.utils import FurthestPointSampler

class CompletionLoss:
    def __init__(self, dataset, loss_func='cd_l1'):

        if loss_func == 'cd_l1':
            metric = chamfer_l1
        elif loss_func == 'cd_l2':
            metric = chamfer_l2
        elif loss_func == 'emd':
            pass
        else:
            raise Exception('loss function {} not supported yet!'.format(loss_func))
        self.metric = metric

        if dataset == 'Completion3D':
            nums_sub_points = [256, 512, 1024]
        elif dataset in ['ShapeNet-34', 'ShapeNet-Unseen21', 'PCN']:
            nums_sub_points = [256, 512, 2048]
        else:
            raise Exception('dataset {} not supported yet!'.format(dataset))

        self.fps_samplers = {}
        for n_points in reversed(nums_sub_points):
            if n_points not in self.fps_samplers:
                self.fps_samplers[n_points] = FurthestPointSampler(n_points)

    def get_loss(self, pcds_pred, partial, gt, test=False):
        loss_all = self.metric(pcds_pred[-1], gt)
        cd_p3 = [loss_all.numpy()[0]]
        for pcd in reversed(pcds_pred[:-1]):
            n_points = pcd.shape[1]
            if n_points == gt.shape[1]:
                cd_loss = self.metric(pcd, gt)
            elif n_points in self.fps_samplers:
                gt = self.fps_samplers[n_points](gt)
                cd_loss = self.metric(pcd, gt)
            else:
                raise Exception('Invalid point cloud list')
            loss_all = loss_all + cd_loss
            if test:
                cd_p3.insert(0, cd_loss)

        if test:
            cd_p3.insert(0, jt.array(0).float())

        return loss_all, cd_p3




