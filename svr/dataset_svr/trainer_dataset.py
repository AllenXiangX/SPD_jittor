import argparse
import yaml
import munch
import jittor as jt
import dataset_svr.dataset_shapenet as dataset_shapenet



def build_dataset(args):
    # Create Datasets
    dataset_train = dataset_shapenet.ShapeNet(args, train=True)
    dataset_test = dataset_shapenet.ShapeNet(args, train=False)

    # Create dataloaders
    dataset_train.set_attrs(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers)
    )

    dataset_test.set_attrs(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers)
    )

    len_dataset = dataset_train.total_len
    len_dataset_test = dataset_test.total_len
    print('Length of train dataset:%d', len_dataset)
    print('Length of test dataset:%d', len_dataset_test)

    return dataset_train, dataset_test

def build_dataset_val(args):

    # Create Datasets
    dataset_test = dataset_shapenet.ShapeNet_val(args, train=False)

    # Create dataloaders
    dataset_test.set_attrs(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers)
    )

    len_dataset_test = dataset_test.total_len
    print('Length of test dataset:%d', len_dataset_test)

    return dataset_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    dataloader_train, dataloader_test = build_dataset(args)
