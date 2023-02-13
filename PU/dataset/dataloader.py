import os
import h5py
import numpy as np
import jittor as jt
from .point_operation import jitter_perturbation_point_cloud, rotate_point_cloud_and_gt, \
    random_scale_point_cloud_and_gt, nonuniform_sampling
from jittor.dataset.dataset import Dataset

def load_h5_data(h5_filename):
    # num_4X_point = 1024
    num_out_point = 1024

    print("h5_filename : ",h5_filename)
    print("use randominput, input h5 file is:", h5_filename)
    f = h5py.File(h5_filename, 'r')
    # input = f['poisson_%d' % num_4X_point][:]
    gt = f['poisson_%d' % num_out_point][:]

    # assert len(input) == len(gt)

    print("Normalization the data")
    data_radius = np.ones(shape=(len(gt)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    # input[:, :, 0:3] = input[:, :, 0:3] - centroid
    # input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    print("total %d samples" % (len(gt)))
    return gt, data_radius


class PUGANDataset(Dataset):
    def __init__(self, h5_filename):
        super().__init__()
        gt, data_radius = load_h5_data(h5_filename)
        self.gt = gt
        self.data_radius = data_radius
        self.set_attrs(
            total_len=len(self.gt),
        )

    def __getitem__(self, index):
        idx = nonuniform_sampling(1024, 256)
        return self.gt[index][idx], self.gt[index], self.data_radius[index]

    def collate_batch(self, batch):
        batch_inp = []
        batch_gt = []
        batch_radius = []

        for sample in batch:
            batch_inp.append(sample[0])
            batch_gt.append(sample[1])
            batch_radius.append(sample[2])

        batch_inp = np.stack(batch_inp, 0)
        batch_gt = np.stack(batch_gt, 0)
        batch_radius = np.array(batch_radius)

        batch_inp = jitter_perturbation_point_cloud(batch_inp, sigma=0.01,
                                                    clip=0.03)
        batch_inp, batch_gt = rotate_point_cloud_and_gt(batch_inp, batch_gt)
        batch_inp, batch_gt, scales = random_scale_point_cloud_and_gt(batch_inp,
                                                                      batch_gt,
                                                                      scale_low=0.8,
                                                                      scale_high=1.2)
        batch_radius = batch_radius * scales

        batch_inp = jt.array(batch_inp).float()
        batch_gt = jt.array(batch_gt).float()
        batch_radius = jt.array(batch_radius).float()

        return batch_inp, batch_gt, batch_radius


def load_lists(dir_gt):
    gt_files = [f for f in os.listdir(dir_gt) if f.endswith('.xyz')]
    gt_list = []
    names_list = []
    for f in gt_files:
        gt_list.append(np.loadtxt(os.path.join(dir_gt, f)))
        names_list.append(f)

    gt_list = np.stack(gt_list, 0)
    return gt_list, names_list




class PUGANTestset(Dataset):
    def __init__(self, path='/data1/xp/data/PUGAN/gt_8192', path_inp='/data1/xp/data/PUGAN/8192_fps_2048'):
        super().__init__()
        gt, file_names = load_lists(path)

        self.gt = gt
        self.file_names = file_names
        self.LEN = len(self.gt)

        self.inp = None
        if path_inp is not None:
            inp, file_names = load_lists(path_inp)
            self.inp = inp

        self.set_attrs(
            total_len=self.LEN
        )


    def __getitem__(self, index):
        gt = self.gt[index]
        file_name = self.file_names[index]

        if self.inp is None:
            idx = nonuniform_sampling(8192, 2048)
            # idx = np.random.choice(8192, 2048, replace=False)
            inp = gt[idx]
        else:
            inp = self.inp[index]

        return inp, gt, file_name

    def collate_batch(self, batch):
        batch_inp = []
        batch_gt = []
        batch_fn = []
        for sample in batch:
            batch_inp.append(sample[0])
            batch_gt.append(sample[1])
            batch_fn.append(sample[2])

        batch_inp = np.stack(batch_inp, 0)
        batch_gt = np.stack(batch_gt, 0)

        batch_inp = jt.array(batch_inp).float()
        batch_gt = jt.array(batch_gt).float()

        return batch_inp, batch_gt, batch_fn