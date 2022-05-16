import argparse
import os
import math
from tabnanny import check
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dataset import Dataset, LabelManager
from filtration import (FilterBlackAndWhite, FilterFocusMeasure, FilterHSV,
                        FilterManager)
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import DenseNet
from torchvision import transforms
from tqdm import tqdm
import json
import time
from tiling.tile_dataset import TilesDataset

from my_model import MyModel
from aws_utils.s3_sagemaker_utils import S3SageMakerUtils


def load_model(checkpoint_dir: str, my_model: MyModel, distributed: bool = False):
    """
    load_model Loads the model using a checkpoint dir if necessary

    :param checkpoint_dir: Filepath to checkpoint directory for mid train saving
    :type checkpoint_dir: str
    :param my_model: MyModel class hosting the PyTorch model
    :type my_model: MyModel
    :param distributed: Determines if distributed learning is occurring, defaults to False
    :type distributed: bool, optional
    :return: Returns the epoch number at which the checkpoint has saved, or double the output model.
    :rtype: int
    """
    if distributed:
        # Initialize the distributed environment.
        world_size = len(hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = hosts.index(current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=dist_backend,
                                rank=host_rank, world_size=world_size)
        print(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), num_gpus))
        local_rank = os.environ["LOCAL_RANK"]
        torch.cuda.set_device(local_rank)
        batch_size //= dist.get_world_size()
        batch_size = max(batch_size, 1)

        my_model.parallel(distributed)

    if not os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint.pth')):
        print("no checkpoint")
        epoch_number = 0
    else:
        epoch_number = my_model.load_checkpoint()

    return epoch_number


def initialize_data(train_dir: str, val_dir, filtration, filtration_cache, label_encoder, distributed=False, val=False, tiled=False, augmentation=False):
    dataset, data_loader = {}, {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 359))
    ])
    dataset['train'] = Dataset(data_dir=train_dir,
                               labels=LabelManager(
                                   train_dir,
                                   label_postprocessor=label_encoder),
                               filtration=filtration,
                               filtration_cache=filtration_cache,
                               augmentation=transform if augmentation else None)
    if val:
        dataset['val'] = Dataset(data_dir=val_dir,
                                 labels=LabelManager(
                                     val_dir, label_postprocessor=label_encoder),
                                 filtration=filtration,
                                 filtration_cache=filtration_cache)
                                 
    if tiled:
        dataset['train'] = TilesDataset(dataset['train'], '/opt/ml/scoring_data.json')
    train_sampler, val_sampler = None, None
    if distributed:
        train_sampler = DistributedSampler(
            dataset['train'],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank())
    if val:
        val_sampler = DistributedSampler(
            dataset['val'],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank())
    else:
        y_train = dataset['train'].get_region_labels_as_list()
        class_sample_count = np.array(
            list(dataset['train'].get_label_distribution().values()))
        weight = 1 / (class_sample_count + 1e-6)  # in case no values, no div_0
        samples_weight = torch.from_numpy(
            np.array([weight[t] for t in y_train])).double()
        train_sampler = WeightedRandomSampler(
            samples_weight, len(samples_weight))

    data_loader['train'] = DataLoader(dataset['train'],
                                      batch_size=batch_size,
                                      sampler=train_sampler,
                                      num_workers=num_workers,
                                      pin_memory=True)
    if val:
        data_loader['val'] = DataLoader(dataset['val'],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       sampler=val_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)
    return dataset, data_loader


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def main():
    """Main"""
    train_dir = SM_CHANNEL_TRAIN
    test_dir = SM_CHANNEL_TEST
    model_dir = SM_MODEL_DIR
    checkpoint_dir = SM_CHECKPOINT_DIR
    n_epochs = num_epochs
    # filtration = None
    filtration = FilterManager(
        filters=[FilterBlackAndWhite(),
                 FilterHSV(),
                 FilterFocusMeasure()])
    session = S3SageMakerUtils()
    filtration_cache = 'filtration_cache.h5'

    try:
        session.download_data('.', 'digpath-cache',
                              f'{UNIQUE_IMAGE_IDENTIFIER}/{filtration_cache}')
    except:
        print('Filtration Cache Download from S3 failed!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(str(device))
    model = DenseNet(growth_rate=growth_rate,
                     block_config=block_config,
                     num_init_features=num_init_features,
                     bn_size=bn_size,
                     drop_rate=drop_rate,
                     num_classes=num_classes).to(device)
    optim = Adam(model.parameters())
    labels = {label: idx for idx, label in enumerate(classes)}

    def label_encoder(x):
        return labels[os.path.basename(x)]

    try:
        session.download_data(
            '/opt/ml/', 'digpath-tilescore', 'digpath-data-new/scoring_data.json')
    except:
        print('No tiles')

    dataset, data_loader = initialize_data(train_dir, test_dir, filtration, filtration_cache, label_encoder,
                                           distributed=False, val=False)

    try:
        session.upload_data(filtration_cache, 'digpath-cache',
                            f'{UNIQUE_IMAGE_IDENTIFIER}')
    except:
        print('Filtration Cache Upload to S3 failed!')

    criterion = nn.CrossEntropyLoss().to(device)
    best_loss_on_test = np.Infinity

    my_model = MyModel(model, criterion, device,
                       checkpoint_dir, model_dir, optim)
    epoch_number = load_model(checkpoint_dir, my_model, distributed=False)
    if epoch_number == n_epochs:
        n_epochs *= 2

    start_time = time.time()
    writer = SummaryWriter('/opt/ml/output/tensorboard/')
    for epoch in (pbar := tqdm(range(epoch_number, n_epochs))):
        pbar.set_description(f'epoch_progress_{epoch}', refresh=True)

        my_model.train_model(data_loader['train'])
        # my_model.eval(data_loader['val'], num_classes)

        all_loss = my_model.all_loss
        writer.add_scalar(f'train/loss', all_loss["train"], epoch_number)
        writer.add_scalar(
            f'train/acc', my_model.all_acc["train"], epoch_number)
        for r in range(num_classes):
            for c in range(num_classes):  # essentially write out confusion matrix
                writer.add_scalar(
                    f'train/{r}{c}', my_model.cmatrix["train"][r][c], epoch_number)
        print('%s ([%d/%d] %d%%), train loss: %.4f' % (timeSince(start_time, (epoch_number + 1) / num_epochs),
                                                       epoch_number + 1, num_epochs, (epoch_number + 1) / num_epochs * 100, all_loss["train"]), end="")

        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        if all_loss["train"] < best_loss_on_test:
            best_loss_on_test = all_loss["train"]

            state = {
                'epoch': epoch + 1,
                'best_loss_on_test': all_loss,
                'in_channels': in_channels,
                'growth_rate': growth_rate,
                'block_config': block_config,
                'num_init_features': num_init_features,
                'bn_size': bn_size,
                'drop_rate': drop_rate,
                'num_classes': num_classes
            }

            my_model.save_checkpoint(state)
    my_model.save_model()
    region, _ = dataset['train'][2]
    print(my_model.diagnose_region(region, labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--growth-rate', type=int, default=32)
    parser.add_argument('--block-config', type=tuple, default=(2, 2, 2, 2))
    parser.add_argument('--num-init-features', type=int, default=64)
    parser.add_argument('--bn-size', type=int, default=4)
    parser.add_argument('--drop-rate', type=int, default=0)
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--classes',
                        type=tuple,
                        default=('Mild', 'Moderate', 'Severe'))
    parser.add_argument('--dist_backend', type=str, default='gloo')

    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int,
                        default=os.environ['SM_NUM_GPUS'])

    args = vars(parser.parse_args())
    dataname = "digpath_supervised"
    SM_CHANNEL_TRAIN = os.getenv('SM_CHANNEL_TRAIN')
    SM_CHANNEL_TEST = os.getenv('SM_CHANNEL_TEST')
    SM_MODEL_DIR = os.getenv('SM_MODEL_DIR')
    SM_CHECKPOINT_DIR = os.getenv('SM_CHECKPOINT_DIR')
    UNIQUE_IMAGE_IDENTIFIER = os.getenv('UNIQUE_IMAGE_IDENTIFIER')

    # number of classes in the data mask that we'll aim to predict
    num_classes = args['num_classes']
    classes = args['classes']
    in_channels = args['in_channels']  # input channel of the data, RGB = 3
    growth_rate = args['growth_rate']
    block_config = args['block_config']
    num_init_features = args['num_init_features']
    bn_size = args['bn_size']
    drop_rate = args['drop_rate']
    batch_size = args['batch_size']
    # currently, this needs to be 224 due to densenet architecture
    patch_size = args['patch_size']
    num_epochs = args['num_epochs']
    num_gpus = args['num_gpus']
    hosts = args['hosts']
    current_host = args['current_host']
    dist_backend = args['dist_backend']
    num_workers = args['num_workers']
    main()
