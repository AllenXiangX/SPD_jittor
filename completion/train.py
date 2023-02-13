# -*- coding: utf-8 -*-
# @Author: XP

import os
import argparse
import open3d
import numpy as np
import jittor as jt

from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import helpers, average_meter, scheduler, yaml_reader, loss_util, misc
from core import builder
from test import test

def set_seed(seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./configs/pcn_cd1.yaml', help='Configuration File')
    args = parser.parse_args()
    return args

def train(config):

    # dataloaders
    train_dataloader = builder.make_dataloader(config, 'train')
    test_dataloader = builder.make_dataloader(config, config.test.split)

    model = builder.make_model(config)

    # out folders
    if not config.train.out_path:
        config.train.out_path = './exp'
    output_dir = os.path.join(config.train.out_path, '%s', datetime.now().isoformat())
    config.train.path_checkpoints = output_dir % 'checkpoints'
    config.train.path_logs = output_dir % 'logs'
    if not os.path.exists(config.train.path_checkpoints):
        os.makedirs(config.train.path_checkpoints)

    # log writers
    train_writer = SummaryWriter(os.path.join(config.train.path_logs, 'train'))
    val_writer = SummaryWriter(os.path.join(config.train.path_logs, 'test'))

    init_epoch = 1
    best_metric = float('inf')
    steps = 0

    if config.train.resume:
        if not os.path.exists(config.train.model_path):
            raise Exception('checkpoints does not exists: {}'.format(config.test.model_path))

        print('Recovering from %s ...' % (config.train.model_path), end='')
        checkpoint = jt.load(config.test.model_path)
        model.load(checkpoint['model'])
        print('recovered!')

        init_epoch = checkpoint['epoch_index']
        best_metric = checkpoint['best_metric']

    optimizer = builder.make_optimizer(config, model)
    scheduler = builder.make_schedular(config, optimizer, last_epoch=init_epoch if config.train.resume else -1)

    multiplier = 1.0
    if config.test.loss_func == 'cd_l1':
        multiplier = 1e3
    elif config.test.loss_func == 'cd_l2':
        multiplier = 1e4
    elif config.test.loss_func == 'emd':
        multiplier = 1e2

    completion_loss = loss_util.CompletionLoss(config.dataset.name, loss_func=config.train.loss_func)
    fps_sampler = loss_util.FurthestPointSampler(2048)

    n_batches = len(train_dataloader)
    avg_meter_loss = average_meter.AverageMeter(['cd_loss'])
    for epoch_idx in range(init_epoch, config.train.epochs):
        avg_meter_loss.reset()
        model.train()

        with tqdm(train_dataloader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):

                if config.dataset.name in ['PCN', 'Completion3D']:
                    partial = jt.array(data['partial_cloud'])
                    gt = jt.array(data['gtcloud'])
                elif config.dataset.name in ['ShapeNet-34', 'ShapeNet-Unseen21']:
                    npoints = config.dataset.n_points
                    gt = jt.array(data)
                    partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                                                          fps_sampler, fixed_points=None)
                    partial = jt.array(partial)


                pcds_pred = model(partial)
                loss_total, losses = completion_loss.get_loss(pcds_pred, partial, gt)

                # optimizer.backward(loss_total)
                optimizer.step(loss_total)

                # from time import time

                # std = time()
                # for i in range(100):


               #  end = time()
                #print('time: {}'.format(end - std))
                # exit(0)

                # losses = [0]
                avg_meter_loss.update(losses)
                n_itr = epoch_idx * n_batches + batch_idx

                losses = [ls*multiplier for ls in losses]
                train_writer.add_scalar('Loss/Batch/cd_loss', losses[0], n_itr)

                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, config.train.epochs, batch_idx + 1, n_batches))
                t.set_postfix(
                    loss='%s' % ['%.4f' % l for l in losses])

        scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0].get('lr'))

        train_writer.add_scalar('Loss/Epoch/cd_Loss', avg_meter_loss.avg(0), epoch_idx)


        cd_eval = test(config, model=model, test_dataloader=test_dataloader, validation=True,
                       epoch_idx=epoch_idx, test_writer=val_writer)

        # Save checkpoints
        if epoch_idx % config.train.save_freq == 0 or cd_eval < best_metric:
            file_name = 'ckpt-best.pkl' if cd_eval < best_metric else 'ckpt-epoch-%03d.pkl' % epoch_idx
            output_path = os.path.join(config.train.path_checkpoints, file_name)
            jt.save({
                'epoch_index': epoch_idx,
                'best_metric': best_metric,
                'model': model.state_dict()
            }, output_path)

            print('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metric:
                best_metric = cd_eval

    train_writer.close()
    val_writer.close()






if __name__ == '__main__':
    args = get_args_from_command_line()
    config = yaml_reader.read_yaml(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.train.gpu)
    jt.flags.use_cuda = 1
    set_seed(config.train.seed)

    train(config)
