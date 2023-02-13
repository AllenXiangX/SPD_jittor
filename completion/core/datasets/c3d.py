import json
import logging
from tqdm import tqdm
from .utils import Compose
from .dataset import Dataset


class C3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.dataset.category_file_path) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, subset)
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud'] if subset == 'test' else ['partial_cloud', 'gtcloud']

        return Dataset({
            'required_items': required_items,
            'shuffle': subset == 'train'
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == 'train':
            return Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.dataset.n_points
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud', 'gtcloud']
                }])
        elif subset == 'val':
            return Compose([{
                'callback': 'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            },{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])


    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                        dc['taxonomy_id'],
                    'model_id':
                        s,
                    'partial_cloud_path':
                        cfg.dataset.partial_points_path % (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                        cfg.dataset.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list
