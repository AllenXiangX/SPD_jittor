import os
import sys
sys.path.append('..')
import time
import math
import argparse
import numpy as np
import jittor as jt
from tqdm.auto import tqdm
from jittor.contrib import concat
from utils.dataset import ShapeNetCore
from utils.misc import seed_all, get_logger, str_list
from models.model_vae import ModelVAE
from evaluation import compute_all_metrics, jsd_between_point_cloud_sets
method = 'SPD'

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max = pc.max(dim=0, keepdims=True)  # (1, 3)
            pc_min = pc.min(dim=0, keepdims=True)  # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--ckpt', type=str,
                    default='')
parser.add_argument('--categories', type=str_list, default=['chair'])
# parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/data/xp/code/diffusion/data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'GEN_%s_%s_%d' % (method, '_'.join(args.categories), int(time.time())))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = jt.load(args.ckpt)
seed_all(args.seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=args.normalize,
)
test_dset.set_attrs(
    batch_size=1,
    num_workers=1
)


# Model
logger.info('Loading model...')
model = ModelVAE(dim_feat=ckpt['args'].latent_dim,
                 args=ckpt['args'],
                 up_factors=[2, 2])


# logger.info(repr(model))

model.load_state_dict(ckpt['state_dict'])

# Reference Point Clouds
ref_pcs = []
for i, data in enumerate(test_dset):

    ref_pcs.append(data['pointcloud'])
ref_pcs = concat(ref_pcs, dim=0)

# Generate Point Clouds
gen_pcs = []
for i in tqdm(range(0, math.ceil(len(test_dset) / args.batch_size)), 'Generate'):
    with jt.no_grad():
        z = jt.randn([args.batch_size, ckpt['args'].latent_dim])

        x = model.sample(z)
        gen_pcs.append(x.detach().cpu())
gen_pcs = concat(gen_pcs, dim=0)[:len(test_dset)]
if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)

# Save
logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), ref_pcs.numpy())
np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())

# Compute metrics
with jt.no_grad():
    results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size,
                                  accelerated_cd=True)
    results = {k: v.item() for k, v in results.items()}
    jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
