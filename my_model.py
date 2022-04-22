from typing import Callable
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
            'train' : torch.zeros(0, dtype=torch.float64).to(device),
            'val' : 0,
        }
        # self.all_loss = {
        #     key: torch.zeros(0, dtype=torch.float64).to(device)
        #     for key in phases
        # }
        self.cmatrix = {key: np.zeros((num_classes, num_classes)) for key in phases}
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = optimizer

    def parallel(self, distributed: bool = True):
        """parallel"""
        if distributed:
            self.model = DDP(self.model)
        elif torch.cuda.device_count() > 1:
            print("Gpu count: {}".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)

    def train_model(self, data_loader: DataLoader):
        """Train Model"""
        self.model.train()
        for ii, (X, label) in enumerate(data_loader):
            # pbar.set_description(f'training_progress_{ii}', refresh=True)
            X = X.to(self.device)
            label = label.type('torch.LongTensor').to(self.device)
            with torch.set_grad_enabled(True):
                prediction = self.model(X.permute(0, 3, 1, 2).float())  # [N, Nclass]
                loss = self.loss_fn(prediction, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.all_loss['train'] = torch.cat((self.all_loss['train'], loss.detach().view(1, -1)))
                self.all_loss['train'] += loss.item()
        self.all_acc['train'] = (self.cmatrix['train'] / (self.cmatrix['train'].sum() + 1e-6)).trace()
        self.all_loss['train'] /= len(data_loader)
        # self.all_loss['train'] = self.all_loss['train'].cpu().numpy().mean()

    def eval(self, data_loader: DataLoader, num_classes: int):
        """Eval"""
        self.model.eval()

        loss_by_file = {}
        cmatrix_by_file = {}
        dataset = data_loader.dataset
        for file_name, label, regions in dataset.iterate_by_file():
            num_regions = dataset.number_of_regions(file_name)
            loss_by_file[file_name] = 0 #torch.zeros(0, dtype=torch.float64).to(self.device)
            cmatrix_by_file[file_name] = np.zeros((num_classes, num_classes))
            for batch in iterate_by_n(regions, data_loader.batch_size, yield_remainder=True):
                for ii, X in enumerate((pbar := tqdm(batch))):
                    pbar.set_description(f'validation_progress_{ii}', refresh=True)
                    X = torch.tensor(X).to(self.device)
                    # label = torch.tensor(list(map(int, label))).to(self.device)
                    label = torch.tensor([label]).to(self.device)
                    with torch.no_grad():
                        # print("""###################################################################\n# Code is broken within this block\n###################################################################""")
                        prediction = self.model(X[None, ...].permute(0, 3, 2, 1).float())  # [N, Nclass]
                        # print('prediction: ', prediction)
                        loss = self.loss_fn(prediction, label)
                        # print('loss: ', loss)
                        p = prediction.detach().cpu().numpy()
                        # print('p: ', p)
                        cpredflat = np.argmax(p, axis=1).flatten()
                        # print('cpredflat: ', cpredflat)
                        yflat = label.cpu().numpy().flatten()
                        # print('yflat: ', yflat)
                        # print("Loss:", loss.shape)
                        # print("Stored Loss:", loss_by_file[file_name])
                        new_loss = loss_by_file[file_name] + loss.item()
                        # print('new_loss: ', new_loss)
                        loss_by_file[file_name] = new_loss #torch.add(loss_by_file[file_name], new_loss)
                        # print('loss_by_file: ', loss_by_file[file_name])
                        cmatrix_by_file[file_name] = np.add(cmatrix_by_file[file_name], confusion_matrix(yflat, cpredflat, labels=range(num_classes)))
                        # print("""###################################################################\n# You survived one more iteration! Good job.\n###################################################################""")
                        # self.all_loss['val'] = torch.cat((self.all_loss['val'], loss.detach().view(1, -1)))
                        # self.cmatrix['val'] = self.cmatrix['val'] + confusion_matrix(yflat, cpredflat, labels=range(num_classes))

            self.all_loss['val'] = loss_by_file[file_name] / num_regions
            self.cmatrix['val'] = cmatrix_by_file[file_name] / num_regions
        self.all_acc['val'] = (self.cmatrix['val'] / self.cmatrix['val'].sum()).trace()
        # self.all_loss['val'] = self.all_loss['val'].cpu().numpy().mean()

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
        print(f'Checkpoint File Loaded - epoch_number: {epoch_number} - loss: {loss}')
        print(f'Resuming training from epoch: {epoch_number + 1}')
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
        region = torch.Tensor(region[None, ::]).permute(0, 3, 1, 2).float().to(self.device)
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