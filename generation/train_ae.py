import os
import sys
sys.path.append('..')
import argparse
import jittor as jt
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from jittor.contrib import concat
from utils.dataset import ShapeNetCore
from utils.misc import seed_all, get_logger, str_list, \
    THOUSAND, get_new_log_dir, CheckpointManager, BlackHole, get_linear_scheduler
from utils.data import get_data_iterator
from utils.transform import RandomRotate
from models.model_ae import ModelAE
method = 'SPD'
from evaluation import EMD_CD

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--log_root', type=str, default='')
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/home/ld_test/xp/data/shapenet.hdf5')
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=96)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--rotate', type=eval, default=True, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=50 * THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=100 * THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])

parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='AE_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
transform = None
if args.rotate:
    transform = RandomRotate(180, ['pointcloud'], axis=1)
logger.info('Transform: %s' % repr(transform))
logger.info('Loading datasets...')
train_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='train',
    scale_mode=args.scale_mode,
    transform=transform,
)
val_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='val',
    scale_mode=args.scale_mode,
    transform=transform,
)

train_dset.set_attrs(
    batch_size=args.train_batch_size,
    num_workers=args.num_workers
)
train_iter = get_data_iterator(train_dset)

val_dset.set_attrs(
    batch_size=args.val_batch_size,
    num_workers=args.num_workers
)
val_loader = val_dset

# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = jt.load(args.resume)
    model = ModelAE(dim_feat=ckpt['args'].latent_dim)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = ModelAE(dim_feat=args.latent_dim)

# Optimizer and scheduler
optimizer = jt.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay
                             )
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate
def train(it):
    # Load data
    batch = next(train_iter)
    x = jt.array(batch['pointcloud'])

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    #loss = model.get_loss(x)
    loss = model.get_loss(x)

    # Backward and optimize
    optimizer.backward(loss)

    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f ' % (it, loss.item()))
    writer.add_scalar('train/loss', loss.item(), it)
    writer.add_scalar('train/lr', optimizer.param_groups[0].get('lr'), it)
    writer.flush()


def validate_loss(it):
    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref = jt.array(batch['pointcloud'])
        shift = jt.array(batch['shift'])
        scale = jt.array(batch['scale'])
        with jt.no_grad():
            model.eval()
            code = model.encode(ref)
            recons = model.decode(code)

        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    all_refs = concat(all_refs, dim=0)
    all_recons = concat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size, accelerated_cd=True)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()

    logger.info('[Val] Iter %04d | CD %.6f | EMD %.6f  ' % (it, cd, emd))
    writer.add_scalar('val/cd', cd, it)
    writer.add_scalar('val/emd', emd, it)
    writer.flush()

    return cd


def validate_inspect(it):
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = jt.array(batch['pointcloud'])
        model.eval()
        code = model.encode(x)
        recons = model.decode(code).detach()

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break  # Inspect only 5 batch

    writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds].numpy(), global_step=it)
    writer.flush()


# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with jt.no_grad():
                cd_loss = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'i_iter': it
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')