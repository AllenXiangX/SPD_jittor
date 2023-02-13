import os
import yaml
import munch
import random
import logging
import argparse
import jittor as jt
from tqdm import tqdm
from utils.train_utils import AverageValueMeter
from utils.model_utils import calc_cd
from dataset_svr.trainer_dataset import build_dataset_val
from models.model_svr import ModelSVR

import warnings
warnings.filterwarnings("ignore")

def val():
    
    dataloader_test = build_dataset_val(args)

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    jt.set_global_seed(seed)

    net = ModelSVR(
        dim_feat=args.model.dim_feat,
        num_pc=args.model.num_pc,
        num_p0=args.model.num_p0,
        radius=args.model.radius,
        bounding=args.model.bounding,
        up_factors=args.model.up_factors,
    )
    
    ckpt = jt.load(args.load_model)
    net.load_state_dict(ckpt['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.load_model)

    net.eval()

    logging.info('Testing...')

    test_loss_l1 = AverageValueMeter()
    test_loss_l2 = AverageValueMeter()

    with tqdm(dataloader_test) as t:
        for i, data in enumerate(t):
            with jt.no_grad():
        
                images = jt.array(data['image'])
                gt = jt.array(data['points'])

                batch_size = gt.shape[0]
                
                pred_points = net(images)[-1]

                loss_p, loss_t = calc_cd(pred_points, gt)

                cd_l1_item = loss_p.item()  # torch.sum(loss_p).item() / batch_size
                cd_l2_item = loss_t.item()  # torch.sum(loss_t).item() / batch_size
                test_loss_l1.update(cd_l1_item, images.shape[0])
                test_loss_l2.update(cd_l2_item, images.shape[0])

    print('cd_l1 %f cd_l2 %f' % (test_loss_l1.avg, test_loss_l2.avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    val()
