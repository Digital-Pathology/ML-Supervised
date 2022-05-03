from typing import Callable, Optional
import torch
import os
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import confusion_matrix

from unified_image_reader import Image
from model_manager.util import iterate_by_n
from utils import label_decoder


class MyModel:
    """
     _summary_
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: str, checkpoint_dir: str, model_dir: str, optimizer: Optimizer):
        """
        __init__ _summary_

        :param model: _description_
        :type model: nn.Module
        :param loss_fn: _description_
        :type loss_fn: nn.Module
        :param device: _description_
        :type device: str
        :param checkpoint_dir: _description_
        :type checkpoint_dir: str
        :param model_dir: _description_
        :type model_dir: str
        :param optimizer: _description_
        :type optimizer: Optimizer
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        phases = ["train"]
        num_classes = 3
        self.all_acc = {key: 0 for key in phases}
        self.all_loss = {
            key: torch.zeros(0, dtype=torch.float64).to(device)
            for key in phases
        }
        self.cmatrix = {key: np.zeros(
            (num_classes, num_classes)) for key in phases}
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = optimizer

    def parallel(self, distributed: bool = False):
        """
        parallel _summary_
        """
        if distributed:
            self.model = DDP(self.model)
        elif torch.cuda.device_count() > 1:
            print(f"Gpu count: {torch.cuda.device_count()}")
            self.model = nn.DataParallel(self.model)

    def train_model(self, data_loader: DataLoader):
        """

        :param data_loader: _description_
        :type data_loader: DataLoader
        """
        self.all_loss['train'] = torch.zeros(
            0, dtype=torch.float64).to(self.device)
        self.model.train()
        for ii, (X, label) in enumerate(data_loader):
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
        """
        eval _summary_

        :param data_loader: _description_
        :type data_loader: DataLoader
        :param num_classes: _description_
        :type num_classes: int
        """
        self.model.eval()
        self.all_loss['val'] = torch.zeros(
            0, dtype=torch.float64).to(self.device)
        for ii, (X, label) in enumerate((pbar := tqdm(data_loader))):
            pbar.set_description(f'validation_progress_{ii}', refresh=True)
            X = X.to(self.device)
            label = torch.tensor(list(map(int, label))).to(self.device)
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

    def save_model(self, filepath: Optional[str] = None):
        """
        save_model _summary_
        """
        print("Saving the model.")
        path = filepath or os.path.join(self.model_dir, 'model.pth')
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        torch.save(self.model.cpu().state_dict(), path)

    def save_checkpoint(self, state: dict):
        """
        save_checkpoint _summary_

        :param state: _description_
        :type state: dict
        """
        path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        print("Saving the Checkpoint: {}".format(path))
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **state
        }, path)

    def load_checkpoint(self):
        """
        load_checkpoint _summary_

        :return: _description_
        :rtype: _type_
        """
        print("--------------------------------------------")
        print("Checkpoint file found!")
        path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        print("Loading Checkpoint From: {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_number = checkpoint['epoch']
        loss = checkpoint['best_loss_on_test']
        print("Checkpoint File Loaded - epoch_number: {} - loss: {}".format(epoch_number, loss))
        print('Resuming training from epoch: {}'.format(epoch_number + 1))
        print("--------------------------------------------")
        return epoch_number

    def load_model(self, filepath: Optional[str] = None):
        """
        load_model _summary_
        """
        path = filepath or os.path.join(self.model_dir, 'model.pth')
        checkpoint = torch.load(path)
        self.parallel()
        self.model.load_state_dict(checkpoint)

    def diagnose_region(self, region, labels: dict = None):
        """
        diagnose_region _summary_

        :param region: _description_
        :type region: _type_
        :param labels: _description_, defaults to None
        :type labels: dict, optional
        :return: _description_
        :rtype: _type_
        """
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
        """
        diagnose_wsi _summary_

        :param file_path: _description_
        :type file_path: str
        :param aggregate: _description_
        :type aggregate: Callable
        :param classes: _description_
        :type classes: tuple
        :param labels: _description_, defaults to None
        :type labels: dict, optional
        :return: _description_
        :rtype: _type_
        """
        region_classifications = {}
        for i, region in enumerate(Image(file_path)):
            region = region.to(self.device)
            self.model.eval()
            pred = self.diagnose_region(region, labels)
            region_classifications[i] = pred
        return aggregate(region_classifications, classes)
