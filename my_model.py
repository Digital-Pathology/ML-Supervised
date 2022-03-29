from typing import Callable
import torch
import os
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from unified_image_reader import Image
from utils import label_decoder


class MyModel:
    """
    Model
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: str, checkpoint_dir: str, model_dir: str, optimizer: Optimizer):
        """Init"""
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

    def eval(self, data_loader: DataLoader, num_classes: int):
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
        """Diagnose a region"""
        self.model = self.model.to(self.device)
        region = torch.Tensor(region[None, ::]).permute(
            0, 3, 1, 2).float().to(self.device)
        output = self.model(region).to(self.device)
        output = output.detach().squeeze().cpu().numpy()
        pred = np.argmax(output)
        if labels is not None:
            pred = label_decoder(labels, pred)
        return pred

    def diagnose_wsi(self, file_path: str, aggregate: Callable, classes: tuple, labels: dict = None):
        """Diagnose a WSI"""
        region_classifications = {}
        for i, region in enumerate(Image(file_path)):
            region = region.to(self.device)
            self.model.eval()
            pred = self.diagnose_region(region, labels)
            region_classifications[i] = pred
        return aggregate(region_classifications, classes)