import argparse
import os
from tabnanny import check

import numpy as np
import torch
import logging
import torch.distributed as dist
from dataset import Dataset, LabelManager
from filtration import (FilterBlackAndWhite, FilterFocusMeasure, FilterHSV,
                        FilterManager)
from unified_image_reader import Image
# from models import UNet
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MyModel:
    """
    Model
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: str, checkpoint_dir: str, model_dir: str, optimizer: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        phases = ["train", "val"]
        num_classes = 3
        self.all_acc = {key: 0 for key in phases}
        self.all_loss = {
            key: torch.zeros(0, dtype=torch.float64).to(device)
            for key in phases
        }
        self.cmatrix = {key: np.zeros((num_classes, num_classes)) for key in phases}
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = optimizer

    def parallel(self):
        """parallel"""
        if torch.cuda.device_count() > 1:
            print("Gpu count: {}".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)

    def train_model(self, data_loader: DataLoader):
        """Train Model"""
        self.model.train()
        for ii, (X, label) in enumerate((pbar := tqdm(data_loader))):
            pbar.set_description(f'training_progress_{ii}', refresh=True)
            X = X.to(self.device)
            label = label.type('torch.LongTensor').to(self.device)
            with torch.set_grad_enabled(True):
                prediction = self.model(X.permute(0, 3, 1,
                                                  2).float())  # [N, Nclass]
                loss = self.loss_fn(prediction, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.all_loss['train'] = torch.cat(
                    (self.all_loss['train'], loss.detach().view(1, -1)))
        self.all_acc['train'] = (self.cmatrix['train'] /
                                 (self.cmatrix['train'].sum() + 1e-6)).trace()
        self.all_loss['train'] = self.all_loss['train'].cpu().numpy().mean()

    def eval(self, data_loader: DataLoader):
        """Eval"""
        self.model.eval()
        for ii, (X, label) in enumerate((pbar := tqdm(data_loader))):
            pbar.set_description(f'validation_progress_{ii}', refresh=True)
            X = X.to(self.device)
            label = torch.tensor(list(map(lambda x: int(x),
                                          label))).to(self.device)
            with torch.no_grad():
                prediction = self.model(X.permute(0, 3, 1,
                                                  2).float())  # [N, Nclass]
                loss = self.loss_fn(prediction, label)
                p = prediction.detach().cpu().numpy()
                cpredflat = np.argmax(p, axis=1).flatten()
                yflat = label.cpu().numpy().flatten()
                self.all_loss['val'] = torch.cat(
                    (self.all_loss['val'], loss.detach().view(1, -1)))
                self.cmatrix['val'] = self.cmatrix['val'] + \
                    confusion_matrix(yflat, cpredflat,
                                     labels=range(num_classes))
        self.all_acc['val'] = (self.cmatrix['val'] /
                               self.cmatrix['val'].sum()).trace()
        self.all_loss['val'] = self.all_loss['val'].cpu().numpy().mean()

    def save_model(self):
        """Save Model"""
        print("Saving the model.")
        path = os.path.join(self.model_dir, 'model.pth')
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        torch.save(self.model.cpu().state_dict(), path)

    def save_checkpoint(self, state: dict):
        """Save Checkpoint"""
        path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        print("Saving the Checkpoint: {}".format(path))
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **state
        }, path)

    def load_checkpoint(self):
        """Load Checkpoint"""
        print("--------------------------------------------")
        print("Checkpoint file found!")
        path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        print("Loading Checkpoint From: {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_number = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Checkpoint File Loaded - epoch_number: {} - loss: {}".format(epoch_number, loss))
        print('Resuming training from epoch: {}'.format(epoch_number + 1))
        print("--------------------------------------------")
        return epoch_number

    def load_model(self):
        """Load Model"""
        path = os.path.join(self.model_dir, 'model.pth')
        checkpoint = torch.load(path)
        self.parallel()
        self.model.load_state_dict(checkpoint)

    def diagnose_region(self, region, labels: dict = None):
        self.model = self.model.to(self.device)
        region = torch.Tensor(region[None, ::]).permute(
            0, 3, 1, 2).float().to(self.device)
        output = self.model(region).to(self.device)
        output = output.detach().squeeze().cpu().numpy()
        pred = np.argmax(output)
        if labels is not None:
            pred = label_decoder(labels, pred)
        return pred

    def diagnose_wsi(self, file_path: str, aggregate, labels: dict = None):
        region_classifications = {}
        for i, region in enumerate(Image(file_path)):
            region = region.to(self.device)
            self.model.eval()
            pred = self.diagnose_region(region, labels)
            region_classifications[i] = pred
        return aggregate(region_classifications)


def label_decoder(labels: dict, x: int):
    return list(labels.keys())[list(labels.values()).index(x)]


def plurality_vote(region_classifications: dict):
    votes = {c: 0 for c in classes}
    for c in region_classifications.values():
        votes[c] += 1

    return votes[max(votes, key=votes.get)]


def load_model(checkpoint_dir: str, my_model: MyModel, num_epochs: int, distributed: bool = False):
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

    my_model.parallel()

    if not os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint.pth')):
        epoch_number = 0
    else:
        epoch_number = my_model.load_checkpoint()

    if epoch_number == num_epochs:
        num_epochs = 2 * num_epochs
        my_model.load_model()
    return epoch_number


def main():
    """Main"""
    train_dir = SM_CHANNEL_TRAIN
    test_dir = SM_CHANNEL_TEST
    model_dir = SM_MODEL_DIR
    checkpoint_dir = SM_CHECKPOINT_DIR
    filtration = None
    # filtration = FilterManager(
    #     filters=[FilterBlackAndWhite(),
    #              FilterHSV(),
    #              FilterFocusMeasure()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(str(device))
    model = DenseNet(growth_rate=growth_rate,
                     block_config=block_config,
                     num_init_features=num_init_features,
                     bn_size=bn_size,
                     drop_rate=drop_rate,
                     num_classes=num_classes).to(device)
    optim = Adam(model.parameters())
    dataset = {}
    dataLoader = {}
    labels = {label: idx for idx, label in enumerate(classes)}
    def label_encoder(x): return labels[os.path.basename(x)]
    dataset['train'] = Dataset(data_dir=train_dir,
                               labels=LabelManager(
                                   train_dir,
                                   label_postprocessor=label_encoder),
                               filtration=filtration)
    dataLoader['train'] = DataLoader(dataset['train'],
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
    print(f'train dataset region counts: {dataset["train"]._region_counts}')
    dataset['val'] = Dataset(data_dir=test_dir,
                             labels=LabelManager(
                                 test_dir, label_postprocessor=label_encoder),
                             filtration=filtration)

    dataLoader['val'] = DataLoader(dataset['val'],
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)
    print(f"val dataset size:\t{len(dataset['val'])}")
    print(f'val dataset region counts: {dataset["val"]._region_counts}')
    criterion = nn.CrossEntropyLoss().to(device)
    best_loss_on_test = np.Infinity

    my_model = MyModel(model, criterion, device, checkpoint_dir, model_dir, optim)
    epoch_number = load_model(checkpoint_dir, my_model, num_epochs, distributed=False)

    for epoch in (pbar := tqdm(range(epoch_number, num_epochs))):
        pbar.set_description(f'epoch_progress_{epoch}', refresh=True)

        my_model.train_model(dataLoader['train'])
        my_model.eval(dataLoader['val'])

        all_loss = my_model.all_loss

        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]

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
    region, label = dataset['val'][2]
    print(my_model.diagnose_region(region, labels))
    print(labels, label_decoder(label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--growth-rate', type=int, default=32)
    parser.add_argument('--block-config', type=tuple, default=(2, 2, 2, 2))
    parser.add_argument('--num-init-features', type=int, default=64)
    parser.add_argument('--bn-size', type=int, default=4)
    parser.add_argument('--drop-rate', type=int, default=0)
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--train-labels', type=str, default='train_labels.csv')
    parser.add_argument('--test-labels', type=str, default='test_labels.csv')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--classes',
                        type=tuple,
                        default=('Mild', 'Moderate', 'Severe'))
    parser.add_argument('--dist_backend', type=str, default='gloo')

    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])


    args = vars(parser.parse_args())
    dataname = "digpath_supervised"
    SM_CHANNEL_TRAIN = os.getenv('SM_CHANNEL_TRAIN')
    SM_CHANNEL_TEST = os.getenv('SM_CHANNEL_TEST')
    SM_MODEL_DIR = os.getenv('SM_MODEL_DIR')
    SM_CHECKPOINT_DIR = os.getenv('SM_CHECKPOINT_DIR')
    # SM_CHANNEL_TRAIN = "/workspaces/dev-container/ML-Supervised/input/train"
    # SM_CHANNEL_TEST = "/workspaces/dev-container/ML-Supervised/input/test"
    # SM_OUTPUT_DIR = "/workspaces/dev-container/ML-Supervised/output"
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
