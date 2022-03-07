from types import new_class
from unicodedata import decimal
from dataset import Dataset
from filtration import FilterManager, FilterBlackAndWhite, FilterHSV
from model_manager_for_web_app import ModelManager, ManagedModel
from tqdm import tqdm
import numpy as np
import torch
import os
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.models import DenseNet
# from models import UNet
from sklearn.metrics import confusion_matrix


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
parser.add_argument('--edge-weight', type=float, default=1.0)

args = vars(parser.parse_args())

dataname = "digpath_supervised"
SM_CHANNEL_TRAIN = os.getenv('SM_CHANNEL_TRAIN')
SM_CHANNEL_TEST = os.getenv('SM_CHANNEL_TEST')
num_classes = args['num_classes']  # number of classes in the data mask that we'll aim to predict
in_channels = args['in_channels']  # input channel of the data, RGB = 3
growth_rate = args['growth_rate']
block_config = args['block_config']
num_init_features = args['num_init_features']
bn_size = args['bn_size']
drop_rate = args['drop_rate']
batch_size = args['batch_size']
patch_size = args['patch_size']  # currently, this needs to be 224 due to densenet architecture
num_epochs = args['num_epochs']
train_labels = f'{os.getcwd()}/{args["train_labels"]}'
test_labels = f'{os.getcwd()}/{args["test_labels"]}'
output_dir = f'{os.getcwd()}/output'
num_workers = args['num_workers']
edge_weight = args['edge_weight']
phases = ["train", 'val']  # how many phases did we create databases for?
# when should we do valiation? note that validation is *very* time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch
validation_phases = ['val']
# additionally, using simply [], will skip validation entirely, drastically speeding things up


class MyManagedModel(ManagedModel):
    def __init__(self, model, loss_fn, device, all_acc, all_loss, cmatrix):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.all_acc = all_acc
        self.all_loss = all_loss
        self.cmatrix = cmatrix

    def train_model(self, optimizer, data_loader):
        self.model.train()
        for ii, (X, label) in enumerate(data_loader):
            X = X.to(self.device)
            label = torch.tensor(list(map(lambda x: int(x), label))).to(self.device)
            with torch.set_grad_enabled(True):
                prediction = self.model(X.permute(0, 3, 1, 2).float())  # [N, Nclass]
                loss = self.loss_fn(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.all_loss['train'] = torch.cat((self.all_loss['train'], loss.detach().view(1, -1)))
        self.all_acc['train'] = (self.cmatrix['train'] / self.cmatrix['train'].sum()).trace()
        self.all_loss['train'] = self.all_loss['train'].cpu().numpy().mean()
    
    def eval(self, data_loader):
        self.model.eval()
        for ii, (X, label) in enumerate(data_loader):
            X = X.to(self.device)
            label = torch.tensor(list(map(lambda x: int(x), label))).to(self.device)
            with torch.no_grad():
                prediction = self.model(X.permute(0, 3, 1, 2).float())  # [N, Nclass]
                loss = self.loss_fn(prediction, label)
                p = prediction.detach().cpu().numpy()
                cpredflat = np.argmax(p, axis=1).flatten()
                yflat = label.cpu().numpy().flatten()

                self.all_loss['val'] = torch.cat((self.all_loss['val'], loss.detach().view(1, -1)))
                self.cmatrix['val'] = self.cmatrix['val'] + confusion_matrix(yflat, cpredflat, labels=range(num_classes))
        self.all_acc['val'] = (self.cmatrix['val'] / self.cmatrix['val'].sum()).trace()
        self.all_loss['val'] = self.all_loss['val'].cpu().numpy().mean()
    
    def diagnose(self, region_stream):
        votes = {0:0, 1:0, 2:0}
        key = {0: 'MILD', 1: 'Moderate', 2: 'Severe'}
        for region in region_stream:
            self.model.eval()
            output = self.model(region[None, ::].to(self.device))
            output = output.detach().squeeze().cpu().numpy()
            votes[np.argmax(output)] += 1
        return key[max(votes, key=votes.get)] # key with max value


def main():
    train_dir = SM_CHANNEL_TRAIN
    test_dir = SM_CHANNEL_TEST
    filtration = None # FilterManager(filters=[FilterBlackAndWhite(), FilterHSV()])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                     num_init_features=num_init_features,
                     bn_size=bn_size,
                     drop_rate=drop_rate,
                     num_classes=num_classes).to(device)
    optim = Adam(model.parameters())
    dataset = {}
    dataLoader = {}
    dataset['train'] = Dataset(
        data_dir=train_dir, labels=train_labels, filtration=filtration)
    dataLoader['train'] = DataLoader(dataset['train'], batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"train dataset size:\t{len(dataset['train'])}")
    dataset['val'] = Dataset(
        data_dir=test_dir, labels=test_labels, filtration=filtration)
    dataLoader['val'] = DataLoader(dataset['val'], batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"val dataset size:\t{len(dataset['val'])}")
    criterion = nn.CrossEntropyLoss()

    best_loss_on_test = np.Infinity
    edge_weight = torch.tensor(edge_weight).to(device)
    manager = ModelManager(output_dir)
    for epoch in tqdm(range(num_epochs)):
        # zero out epoch based performance variables
        all_acc = {key: 0 for key in phases}
        # keep this on GPU for greatly improved performance
        all_loss = {key: torch.zeros(0).to(device) for key in phases}
        cmatrix = {key: np.zeros((num_classes, num_classes)) for key in phases}

        my_model = MyManagedModel(model, criterion, device, all_acc, all_loss, cmatrix)
        my_model.train_model(optim, dataLoader['train'])
        my_model.eval(dataLoader['val'])

        all_acc, all_loss, cmatrix = my_model.all_acc, my_model.all_loss, my_model.cmatrix

        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]
            print("  **")
            state = {'epoch': epoch + 1,
                     'model_dict': model.state_dict(),
                     'optim_dict': optim.state_dict(),
                     'best_loss_on_test': all_loss,
                     'in_channels': in_channels,
                     'growth_rate': growth_rate,
                     'block_config': block_config,
                     'num_init_features': num_init_features,
                     'bn_size': bn_size,
                     'drop_rate': drop_rate,
                     'num_classes': num_classes}

            # torch.save(state, f"{dataname}_densenet_best_model.pth")
            manager.save_model(model_name=f"{dataname}_densenet_best_model", model=model, model_info=state, overwrite_model = True)
        else:
            print("")
    diagnose_example(model, manager, dataset, device)


def diagnose_example(model, manager, dataset, device):
    img, label, _ = dataset["val"][2]
    manager.load_model(f"{dataname}_densenet_best_model")
    model.load_state_dict(manager.get_model_info(f"{dataname}_densenet_best_model")['model_dict'])
    output = model(img[None, ::].to(device))
    output = output.detach().squeeze().cpu().numpy()
    print(f"True class:{label}")
    print(f"Predicted class:{np.argmax(output)}")


# def train_model(model, optimizer, loss_fn, data_loader, device, all_acc, all_loss, cmatrix):
#     model.train()
#     for ii, (X, label) in enumerate(data_loader):
#         X = X.to(device)
#         label = torch.tensor(list(map(lambda x: int(x), label))).to(device)
#         with torch.set_grad_enabled(True):
#             prediction = model(X.permute(0, 3, 1, 2).float())  # [N, Nclass]
#             loss = loss_fn(prediction, label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             all_loss['train'] = torch.cat((all_loss['train'], loss.detach().view(1, -1)))
#     all_acc['train'] = (cmatrix['train'] / cmatrix['train'].sum()).trace()
#     all_loss['train'] = all_loss['train'].cpu().numpy().mean()
#     return all_acc, all_loss, cmatrix


# def test_model(model, loss_fn, data_loader, device, all_acc, all_loss, cmatrix):
#     model.eval()
#     for ii, (X, label) in enumerate(data_loader):
#         X = X.to(device)
#         label = torch.tensor(list(map(lambda x: int(x), label))).to(device)
#         with torch.no_grad():
#             prediction = model(X.permute(0, 3, 1, 2).float())  # [N, Nclass]
#             loss = loss_fn(prediction, label)
#             p = prediction.detach().cpu().numpy()
#             cpredflat = np.argmax(p, axis=1).flatten()
#             yflat = label.cpu().numpy().flatten()

#             all_loss['val'] = torch.cat((all_loss['val'], loss.detach().view(1, -1)))
#             cmatrix['val'] = cmatrix['val'] + confusion_matrix(yflat, cpredflat, labels=range(num_classes))
#     all_acc['val'] = (cmatrix['val'] / cmatrix['val'].sum()).trace()
#     all_loss['val'] = all_loss['val'].cpu().numpy().mean()
#     return all_acc, all_loss, cmatrix


if __name__ == "__main__":
    main()
