import os
import numpy as np
from jittor.dataset.dataset import Dataset
from .utils import IO

class ShapeNet(Dataset):
    def __init__(self, config, subset):
        super().__init__()
        self.data_root = config.dataset.data_path
        self.pc_path = config.dataset.pc_path
        self.subset = subset
        self.npoints = config.dataset.n_points
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

        self.set_attrs(
            total_len=len(self.file_list)
        )

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        data = self.pc_norm(data)

        return sample['taxonomy_id'], sample['model_id'], data

