import os
import sys
sys.path.append('..')
import numpy as np
import jittor as jt
from tqdm import tqdm
from loss_functions import chamfer_unidirectional_l2

def normalize_point_cloud(pc):
    """
    pc: tensor [N, P, 3]
    """
    centroid = jt.mean(pc, dim=1, keepdims=True)
    pc = pc-centroid
    furthest_distance = jt.max(jt.sqrt(jt.sum(pc**2, dim=-1, keepdims=True)), dim=1, keepdims=True)[0]
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def load_lists(dir_pred, dir_gt):
    gt_files = [f for f in os.listdir(dir_gt) if f.endswith('xyz')]
    pred_list = []
    gt_list = []
    for f in gt_files:
        pred_list.append(np.loadtxt(os.path.join(dir_pred, f)))
        gt_list.append(np.loadtxt(os.path.join(dir_gt, f)))
    return pred_list, gt_list

def evaluate(pred_list, gt_list, device='cuda:0'):
    n = len(pred_list)
    total_cd = 0
    total_hd = 0
    for i in tqdm(range(n)):
        pred = jt.array(pred_list[i]).float().unsqueeze(0)
        gt = jt.from_numpy(gt_list[i]).float().unsqueeze(0)

        pred = normalize_point_cloud(pred)[0]
        gt = normalize_point_cloud(gt)[0]

        d1 = chamfer_unidirectional_l2(pred, gt)
        d2 = chamfer_unidirectional_l2(gt, pred)
        d1 = d1.squeeze(0).numpy()
        d2 = d2.squeeze(0).numpy()

        hd_value = np.max(np.amax(d1, axis=0) + np.amax(d2, axis=0))

        total_cd += (np.mean(d1) + np.mean(d2))
        total_hd += hd_value

    print('avg_cd: ', total_cd / n)
    print('avg_hd: ', total_hd / n)

def evaluate_tensor(pred_list, gt_list):
    """Evaluate batched and normalized predictions and ground truths,
    """
    n = len(pred_list)
    total_cd = 0
    total_hd = 0
    for i in tqdm(range(n)):
        pred = pred_list[i: i+1]
        gt = gt_list[i: i+1]

        pred = normalize_point_cloud(pred)[0]
        gt = normalize_point_cloud(gt)[0]

        d1 = chamfer_unidirectional_l2(pred, gt)
        d2 = chamfer_unidirectional_l2(gt, pred)
        d1 = d1.squeeze(0).numpy()
        d2 = d2.squeeze(0).numpy()

        hd_value = np.max(np.amax(d1, axis=0) + np.amax(d2, axis=0))

        total_cd += np.mean(d1) + np.mean(d2)
        total_hd += hd_value

    avg_cd = total_cd / n
    avg_hd = total_hd / n
    # print('avg_cd: ', avg_cd)
    # print('avg_hd: ', avg_hd)
    return avg_cd, avg_hd

if __name__ == '__main__':
    p_list, g_list = load_lists('/data1/xp/PUGAN/data/test/groundtruth/output', '/data1/xp/PUGAN/data/test/groundtruth')
    p_list = jt.array(np.stack(p_list, 0)).float()
    g_list = jt.array(np.stack(g_list, 0)).float()


    g_list, c, f = normalize_point_cloud(g_list)
    p_list = (p_list - c) / f

    cd, hd = evaluate_tensor(p_list, g_list)
    print('cd: ', cd, 'hd: ', hd)

    # evaluate(p_list, g_list, 'cuda:8')
