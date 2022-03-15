from types import new_class
from unicodedata import decimal
from dataset import Dataset
# from filtration import FilterManager, FilterBlackAndWhite, FilterHSV
from model_manager import ModelManager
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

args = vars(parser.parse_args())

dataname = "digpath_supervised"
SM_CHANNEL_TRAIN = os.getenv('SM_CHANNEL_TRAIN')
SM_CHANNEL_TEST = os.getenv('SM_CHANNEL_TEST')
SM_OUTPUT_DIR = os.getenv('SM_OUTPUT_DIR')
# number of classes in the data mask that we'll aim to predict
num_classes = args['num_classes']
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
train_labels = f'{os.getcwd()}/{args["train_labels"]}'
test_labels = f'{os.getcwd()}/{args["test_labels"]}'
num_workers = args['num_workers']
phases = ["train", 'val']  # how many phases did we create databases for?
# when should we do valiation? note that validation is *very* time consuming, so as opposed to doing for both training and validation, we do it only for validation at the end of the epoch
validation_phases = ['val']
# additionally, using simply [], will skip validation entirely, drastically speeding things up


class MyModel:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: str, all_acc: dict, all_loss: dict, cmatrix: dict):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.all_acc = all_acc
        self.all_loss = all_loss
        self.cmatrix = cmatrix
    def init_log(self, file):
        self.file = file
    def train_model(self, optimizer: torch.optim.Optimizer, data_loader: DataLoader):
        self.file.write("########################   SETTING TO TRAIN MODE!  ########################")
        print("########################   SETTING TO TRAIN MODE!  ########################")
        self.model.train()
        for ii, (X, label) in (pbar := enumerate(tqdm(data_loader))):
            pbar.set_description(f'training_progress_{ii}', refresh=True)
            self.file.write("########################   PUSHING TO DEVICE!  ########################")
            print("########################   PUSHING TO DEVICE!  ########################")
            X, label = X.to(self.device), label.to(self.device)
            label = torch.tensor(
                list(map(lambda x: int(x), label))).to(self.device)
            with torch.set_grad_enabled(True):
                self.file.write("########################   GENERATING OUTPUT!  ########################")
                print("########################   GENERATING OUTPUT!  ########################")
                prediction = self.model(
                    X.permute(0, 3, 1, 2).float())  # [N, Nclass]
                self.file.write("########################   COMPUTING LOSS!  ########################")
                print("########################   COMPUTING LOSS!  ########################")
                loss = self.loss_fn(prediction, label)
                self.file.write("########################   ZERO GRAD!  ########################")
                print("########################   ZERO GRAD!  ########################")
                optimizer.zero_grad()
                self.file.write("########################   BACKPROPOGATION!  ########################")
                print("########################   BACKPROPOGATION!  ########################")
                loss.backward()
                self.file.write("########################   OPTIMIZATION!  ########################")
                print("########################   OPTIMIZATION!  ########################")
                optimizer.step()

                self.file.write("########################   TRAINING LOSS STORAGE!  ########################")
                print("########################   TRAINING LOSS STORAGE!  ########################")
                self.all_loss['train'] = torch.cat(
                    (self.all_loss['train'], loss.detach().view(1, -1)))
        self.file.write("########################   TRAINING ACCURACY!  ########################")
        print("########################   TRAINING ACCURACY!  ########################")
        self.all_acc['train'] = (
            self.cmatrix['train'] / self.cmatrix['train'].sum()).trace()
        self.all_loss['train'] = self.all_loss['train'].cpu().numpy().mean()

    def eval(self, data_loader: DataLoader):
        self.file.write("########################   SETTING TO EVALUATION MODE!  ########################")
        print("########################   SETTING TO EVALUATION MODE!  ########################")
        self.model.eval()
        for ii, (X, label) in (pbar := enumerate(tqdm(data_loader))):
            pbar.set_description(f'validation_progress_{ii}', refresh=True)
            self.file.write("########################   PUSHING TO DEVICE!  ########################")
            print("########################   PUSHING TO DEVICE!  ########################")
            X, label = X.to(self.device), label.to(self.device)
            label = torch.tensor(
                list(map(lambda x: int(x), label))).to(self.device)
            with torch.no_grad():
                self.file.write("########################   GENERATING OUTPUT!  ########################")
                print("########################   GENERATING OUTPUT!  ########################")
                prediction = self.model(
                    X.permute(0, 3, 1, 2).float())  # [N, Nclass]
                self.file.write("########################   COMPUTING LOSS!  ########################")
                print("########################   COMPUTING LOSS!  ########################")
                loss = self.loss_fn(prediction, label)
                p = prediction.detach().cpu().numpy()
                cpredflat = np.argmax(p, axis=1).flatten()
                yflat = label.cpu().numpy().flatten()

                self.file.write("########################   EVALUATION LOSS STORAGE!  ########################")
                print("########################   EVALUATION LOSS STORAGE!  ########################")
                self.all_loss['val'] = torch.cat(
                    (self.all_loss['val'], loss.detach().view(1, -1)))
                self.file.write("########################   CONFUSION MATRIX GENERATION!  ########################")
                print("########################   CONFUSION MATRIX GENERATION!  ########################")
                self.cmatrix['val'] = self.cmatrix['val'] + \
                    confusion_matrix(yflat, cpredflat,
                                     labels=range(num_classes))
        self.file.write("########################   EVALUATION ACCURACY!  ########################")
        print("########################   EVALUATION ACCURACY!  ########################")
        self.all_acc['val'] = (self.cmatrix['val'] /
                               self.cmatrix['val'].sum()).trace()
        self.all_loss['val'] = self.all_loss['val'].cpu().numpy().mean()

    def diagnose(self, region_stream: DataLoader):
        votes = {0: 0, 1: 0, 2: 0}
        key = {0: 'MILD', 1: 'Moderate', 2: 'Severe'}
        for ii, region in (pbar := enumerate(tqdm(region_stream))):
            region = region.to(self.device)
            pbar.set_description(f'diagnose_progress_{ii}', refresh=True)
            self.model.eval()
            output = self.model(region[None, ::].to(self.device))
            output = output.detach().squeeze().cpu().numpy()
            votes[np.argmax(output)] += 1
        return key[max(votes, key=votes.get)]  # key with max value


def main(file):
    train_dir = SM_CHANNEL_TRAIN
    test_dir = SM_CHANNEL_TEST
    output_dir = SM_OUTPUT_DIR
    # FilterManager(filters=[FilterBlackAndWhite(), FilterHSV()])
    filtration = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file.write(device)
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
    file.write(f"train dataset size:\t{len(dataset['train'])}")
    file.write(f'train dataset filepaths: {dataset["train"]._region_counts}')
    print(f"train dataset size:\t{len(dataset['train'])}")
    print(f'train dataset filepaths: {dataset["train"]._region_counts}')
    dataset['val'] = Dataset(
        data_dir=test_dir, labels=test_labels, filtration=filtration)
    
    dataLoader['val'] = DataLoader(dataset['val'], batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, pin_memory=True)
    file.write(f"val dataset size:\t{len(dataset['val'])}")
    file.write(f'val dataset filepaths: {dataset["train"]._region_counts}')
    print(f"val dataset size:\t{len(dataset['val'])}")
    print(f'val dataset filepaths: {dataset["val"]._region_counts}')
    criterion = nn.CrossEntropyLoss()

    best_loss_on_test = np.Infinity
    edge_weight = 1.0
    edge_weight = torch.tensor(edge_weight).to(device)
    manager = ModelManager(output_dir)
    file.write("########################   INITIALIZATION COMPLETE!  ########################")
    print("########################   INITIALIZATION COMPLETE!  ########################")
    for epoch in (pbar := tqdm(range(num_epochs))):
        pbar.set_description(f'epoch_progress_{epoch}', refresh=True)
        # zero out epoch based performance variables
        all_acc = {key: 0 for key in phases}
        # keep this on GPU for greatly improved performance
        all_loss = {key: torch.zeros(0).to(device) for key in phases}
        cmatrix = {key: np.zeros((num_classes, num_classes)) for key in phases}

        my_model = MyModel(model, criterion, device,
                           all_acc, all_loss, cmatrix)
        my_model.init_log(file)
        file.write("########################   STARTING TRAINING!  ########################")
        print("########################   STARTING TRAINING!  ########################")

        my_model.train_model(optim, dataLoader['train'])
        file.write("########################   STARTING EVALUATION!  ########################")
        print("########################   STARTING EVALUATION!  ########################")
        my_model.eval(dataLoader['val'])

        all_acc, all_loss, cmatrix = my_model.all_acc, my_model.all_loss, my_model.cmatrix

        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            file.write("########################   SAVING BEST MODEL!  ########################")
            print("########################   SAVING BEST MODEL!  ########################")
            best_loss_on_test = all_loss["val"]
            f.write("  **")
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
            manager.save_model(model_name=f"{dataname}_densenet_best_model",
                               model=model, model_info=state, overwrite_model=True)
        else:
            file.write("")
    
    diagnose_example(model, manager, dataset, device, file)


def diagnose_example(model, manager, dataset, device, file):
    file.write("########################   DIAGNOSING EXAMPLE!  ########################")
    print("########################   DIAGNOSING EXAMPLE!  ########################")
    img, label, _ = dataset["val"][2]
    file.write("########################   LOADING MODEL!  ########################")
    print("########################   LOADING MODEL!  ########################")
    manager.load_model(f"{dataname}_densenet_best_model")
    file.write("########################   LOADING STATE DICT!  ########################")
    print("########################   LOADING STATE DICT!  ########################")
    model.load_state_dict(manager.get_model_info(
        f"{dataname}_densenet_best_model")['model_dict'])
    file.write("########################   GENERATING OUTPUT!  ########################")
    print("########################   GENERATING OUTPUT!  ########################")
    output = model(img[None, ::].to(device))
    output = output.detach().squeeze().cpu().numpy()
    file.write(f"True class: {label}")
    file.write(f"Predicted class: {np.argmax(output)}")


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
    with open("/opt/ml/model/supervised_logfile.txt", 'w+') as f:
        main(f)
