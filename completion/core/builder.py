import sys
import jittor as jt
from jittor import nn
sys.path.append('../..')
from utils.misc import build_lambda_sche
from models.model_completion import SnowflakeNet


def make_dataloader(config, split):
    dataset_name = config.dataset.name

    if split == 'train':
        batch_size = config.train.batch_size
        num_workers = config.train.num_workers
    else:
        batch_size = config.test.batch_size
        num_workers = config.test.num_workers

    if dataset_name == 'PCN':

        from .datasets.pcn import PCNDataLoader
        dataset = PCNDataLoader(config).get_dataset(split)
    elif dataset_name == 'Completion3D':
        from .datasets.c3d import C3DDataLoader
        dataset = C3DDataLoader(config).get_dataset(split)
    elif dataset_name == 'ShapeNet-34' or dataset_name == 'ShapeNet-Unseen21':
        from .datasets.shapenet55 import ShapeNet
        dataset = ShapeNet(config, split)
    else:
        raise Exception('dataset {} not supported yet!'.format(dataset_name))

    # print('dataset.batch_size', dataset.batch_size)
    dataset.set_attrs(
        batch_size=batch_size,
        shuffle=split == 'train',
        num_workers=num_workers,
        drop_last=False)

    return dataset


def make_model(config):
    model = SnowflakeNet(
        dim_feat=config.model.dim_feat,
        num_pc=config.model.num_pc,
        num_p0=config.model.num_p0,
        radius=config.model.radius,
        bounding=config.model.bounding,
        up_factors=config.model.up_factors,
    )

    # if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model).cuda()

    return model


def make_optimizer(config, model):
    opti_config = config.train.optimizer
    if opti_config.type == 'Adam':
        optimizer = nn.Adam(model.parameters(),
                                     **opti_config.kwargs)
    elif opti_config.type == 'AdamW':
        optimizer = nn.AdamW(model.parameters(),
                                      **opti_config.kwargs)

    else:
        raise Exception('optimizer {} not supported yet!'.format(opti_config.type))


    return optimizer


def make_schedular(config, optimizer, last_epoch=-1):
    sche_config = config.train.scheduler
    if sche_config.type == 'StepLR':
        scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=sche_config.kwargs.milestones, gamma=sche_config.kwargs.gamma, last_epoch=last_epoch)
    elif sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    else:
        raise Exception('scheduler {} not supported yet!'.format(sche_config.type))

    return scheduler