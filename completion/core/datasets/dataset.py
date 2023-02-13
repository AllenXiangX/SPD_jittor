import random
import numpy as np
from jittor.dataset.dataset import Dataset as DatasetJT
from .utils import IO


class Dataset(DatasetJT):
    def __init__(self, options, file_list, transforms=None):
        super().__init__()

        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()
        self.set_attrs(
            total_len=len(file_list),
        )


    def collate_batch(self, batch):
        taxonomy_ids = []
        model_ids = []
        data = {}

        for sample in batch:
            taxonomy_ids.append(sample[0])
            model_ids.append(sample[1])
            _data = sample[2]
            for k, v in _data.items():
                if k not in data:
                    data[k] = []
                data[k].append(np.expand_dims(v, 0))

        for k, v in data.items():
            data[k] = np.concatenate(v, 0)

        return taxonomy_ids, model_ids, data

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            # print(file_path)
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data