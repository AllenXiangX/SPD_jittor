import os
import sys
sys.path.append('..')
import argparse
import numpy as np
import jittor as jt
from tqdm import tqdm
from jittor.contrib import concat
from evaluate import evaluate_tensor, normalize_point_cloud
from dataset.dataloader import PUGANTestset
from utils import read_yaml, patch_extraction, FurthestPointSampler
from models.model_pu import ModelPU

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument spd_pu of SPD for point cloud upsampling')
    parser.add_argument('--config', type=str, default='./configs/spd_pu.yaml', help='Configuration File')
    args = parser.parse_args()
    return args

def test(config, model=None, data_loader=None, epoch=0, best_cd=0, path=None):
    inps = []
    preds = []

    preds_back = []
    gts = []
    _patches = []
    patches = []
    patches_1 = []
    patches_2 = []
    file_names = []
    save_files = False
    if model is None:
        model = ModelPU(up_factors=config.model.up_factors)

        print('loading checkpoint from {}...'.format(config.test.model_path), end='')
        ckp = jt.load(config.test.model_path)
        model.loat(ckp['model'])
        print('loaded!')

        test_dataset = PUGANTestset(path=config.dataset.test_gt_path, path_inp=config.dataset.test_input_path)

        test_dataset.set_attrs(
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False
        )
        data_loader = test_dataset

        path = config.test.save_path
        save_files = config.test.save_output
    model.eval()

    for (inp, gt, fns) in tqdm(data_loader):
        inp = jt.array(inp)
        gt = jt.array(gt)
        b, n, _ = gt.shape

        # inp = normalize_point_cloud(inp)[0]
        # inp = random_subsample(gt, 1024)
        # inp = fps_sample(gt, 2048)
        # gt = normalize_point_cloud(gt)[0]
        inp, centroid, furthest_distance = normalize_point_cloud(inp)

        patch_points = patch_extraction(inp, num_per_patch=256, patch_num_ratio=3)

        normalized_patch_points, centroid_patch, furthest_distance_patch = normalize_point_cloud(patch_points)

        with jt.no_grad():
            p1_patch,  p2_patch, _, normalized_upsampled_patch_points = model(normalized_patch_points)# [0]
            # normalized_upsampled_patch_points = model(normalized_upsampled_patch_points)[-1]

        upsampled_patch_points = (normalized_upsampled_patch_points * furthest_distance_patch) + centroid_patch
        upsampled_points = upsampled_patch_points.reshape((b, -1, 3))
        pred = FurthestPointSampler(n)(upsampled_points)

        p1_patch = (p1_patch * furthest_distance_patch) + centroid_patch
        p2_patch = (p2_patch * furthest_distance_patch) + centroid_patch

        pred_back = (pred * furthest_distance) + centroid

        inps.append(inp)
        _patches.append(patch_points)
        patches.append(upsampled_patch_points)
        patches_1.append(p1_patch)
        patches_2.append(p2_patch)
        preds.append(pred)
        preds_back.append(pred_back)
        gts.append(gt)
        file_names.append(fns[0])


    inps = concat(inps, 0)
    _patches = concat(_patches, 0)
    patches = concat(patches, 0)
    patches_1 = concat(patches_1, 0)
    patches_2 = concat(patches_2, 0)
    preds = concat(preds, 0)
    preds_back = concat(preds_back, 0)
    gts = concat(gts, 0)

    cd, hd = evaluate_tensor(preds_back, gts)
    print('Epoch: ', epoch, 'cd: ', cd*1000, 'hd: ', hd*1000)

    if cd < best_cd or save_files:
        inps = inps.numpy()
        _patches = _patches.numpy()
        preds = preds.numpy()
        gts = gts.numpy()
        patches = patches.numpy()
        patches_1 = patches_1.numpy()
        patches_2 = patches_2.numpy()

        save_dir = path + '/results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(save_dir + '/inps.npy', inps)
        np.save(save_dir + '/_patches.npy', _patches)
        np.save(save_dir + '/patches.npy', patches)
        np.save(save_dir + '/patches_1.npy', patches_1)
        np.save(save_dir + '/patches_2.npy', patches_2)
        np.save(save_dir + '/preds.npy', preds)
        np.save(save_dir + '/gts.npy', gts)
        # np.save(save_dir + '/preds_back.npy', preds_back)

        save_dir_individuals = save_dir + '/xyz'
        if not os.path.exists(save_dir_individuals):
            os.makedirs(save_dir_individuals)

        for i, f in enumerate(file_names):
            np.savetxt(os.path.join(save_dir_individuals, f), preds_back[i].numpy())

    return cd, hd

if __name__ == '__main__':
    args = get_args_from_command_line()
    jt.flags.use_cuda = 1
    config = read_yaml(args.config)
    test(config)