import argparse
import os
from types import new_class
from unicodedata import decimal

import numpy as np
import torch
from dataset import Dataset, LabelManager
from filtration import (FilterBlackAndWhite, FilterFocusMeasure, FilterHSV,
                        FilterManager)
from model_manager import ModelManager
# from models import UNet
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from tqdm import tqdm

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

args = vars(parser.parse_args())
dataname = "digpath_supervised"
SM_CHANNEL_TRAIN = os.getenv('SM_CHANNEL_TRAIN')
SM_CHANNEL_TEST = os.getenv('SM_CHANNEL_TEST')
SM_OUTPUT_DIR = os.getenv('SM_OUTPUT_DIR')
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

num_workers = args['num_workers']
phases = ["train", 'val']  # how many phases did we create databases for?
# when should we do valiation? note that validation is *very* time consuming, so as opposed to doing for both training and validation, we do it only for validation at the end of the epoch
validation_phases = ['val']
# additionally, using simply [], will skip validation entirely, drastically speeding things up


class MyModel:

    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: str,
                 all_acc: dict, all_loss: dict, cmatrix: dict):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.all_acc = all_acc
        self.all_loss = all_loss
        self.cmatrix = cmatrix

    def train_model(self, optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader):
        """Train Model"""
        print(
            "########################   SETTING TO TRAIN MODE!  ########################\n"
        )
        self.model.train()
        for ii, (X, label) in enumerate((pbar := tqdm(data_loader))):
            if ii > 1:
                break
            pbar.set_description(f'training_progress_{ii}', refresh=True)
            print(
                "########################   PUSHING TO DEVICE!  ########################\n"
            )
            X = X.to(self.device)
            label = label.type('torch.LongTensor').to(self.device)
            with torch.set_grad_enabled(True):

                print(
                    "########################   GENERATING OUTPUT!  ########################\n"
                )
                prediction = self.model(X.permute(0, 3, 1,
                                                  2).float())  # [N, Nclass]

                print(
                    "########################   COMPUTING LOSS!  ########################\n"
                )
                loss = self.loss_fn(prediction, label)

                print(
                    "########################   ZERO GRAD!  ########################\n"
                )
                optimizer.zero_grad()

                print(
                    "########################   BACKPROPOGATION!  ########################\n"
                )
                loss.backward()

                print(
                    "########################   OPTIMIZATION!  ########################\n"
                )
                optimizer.step()

                print(
                    "########################   TRAINING LOSS STORAGE!  ########################\n"
                )
                self.all_loss['train'] = torch.cat(
                    (self.all_loss['train'], loss.detach().view(1, -1)))

        print(
            "########################   TRAINING ACCURACY!  ########################\n"
        )
        self.all_acc['train'] = (self.cmatrix['train'] /
                                 (self.cmatrix['train'].sum() + 1e-6)).trace()
        self.all_loss['train'] = self.all_loss['train'].cpu().numpy().mean()

    def eval(self, data_loader: DataLoader):
        """Eval"""

        print(
            "########################   SETTING TO EVALUATION MODE!  ########################\n"
        )
        self.model.eval()
        for ii, (X, label) in enumerate((pbar := tqdm(data_loader))):
            if ii > 1:
                break
            pbar.set_description(f'validation_progress_{ii}', refresh=True)

            print(
                "########################   PUSHING TO DEVICE!  ########################\n"
            )
            X = X.to(self.device)
            label = torch.tensor(list(map(lambda x: int(x),
                                          label))).to(self.device)
            with torch.no_grad():

                print(
                    "########################   GENERATING OUTPUT!  ########################\n"
                )
                prediction = self.model(X.permute(0, 3, 1,
                                                  2).float())  # [N, Nclass]

                print(
                    "########################   COMPUTING LOSS!  ########################\n"
                )
                loss = self.loss_fn(prediction, label)
                p = prediction.detach().cpu().numpy()
                cpredflat = np.argmax(p, axis=1).flatten()
                yflat = label.cpu().numpy().flatten()

                print(
                    "########################   EVALUATION LOSS STORAGE!  ########################\n"
                )
                self.all_loss['val'] = torch.cat(
                    (self.all_loss['val'], loss.detach().view(1, -1)))

                print(
                    "########################   CONFUSION MATRIX GENERATION!  ########################\n"
                )
                self.cmatrix['val'] = self.cmatrix['val'] + \
                    confusion_matrix(yflat, cpredflat,
                                     labels=range(num_classes))

        print(
            "########################   EVALUATION ACCURACY!  ########################\n"
        )
        self.all_acc['val'] = (self.cmatrix['val'] /
                               self.cmatrix['val'].sum()).trace()
        self.all_loss['val'] = self.all_loss['val'].cpu().numpy().mean()

    def diagnose(self, region_stream: DataLoader):
        """Diagnose"""
        votes = {0: 0, 1: 0, 2: 0}
        key = {0: 'MILD', 1: 'Moderate', 2: 'Severe'}
        for ii, region in enumerate((pbar := tqdm(region_stream))):
            region = region.to(self.device)
            pbar.set_description(f'diagnose_progress_{ii}', refresh=True)
            self.model.eval()
            output = self.model(region[None, ::].to(self.device))
            output = output.detach().squeeze().cpu().numpy()
            votes[np.argmax(output)] += 1
        return key[max(votes, key=votes.get)]  # key with max value


def main():
    train_dir = SM_CHANNEL_TRAIN
    test_dir = SM_CHANNEL_TEST
    output_dir = SM_OUTPUT_DIR
    # filtration = None
    filtration = FilterManager(
        filters=[FilterBlackAndWhite(),
                 FilterHSV(),
                 FilterFocusMeasure()])
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
    label_encoder = lambda x: labels[os.path.basename(os.path.dirname(x))]
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
    print(f"train dataset size:\t{len(dataset['train'])}")
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
    criterion = nn.CrossEntropyLoss()

    best_loss_on_test = np.Infinity
    edge_weight = 1.0
    edge_weight = torch.tensor(edge_weight).to(device)
    manager = ModelManager(output_dir)

    print(
        "########################   INITIALIZATION COMPLETE!  ########################\n"
    )
    for epoch in (pbar := tqdm(range(num_epochs))):
        pbar.set_description(f'epoch_progress_{epoch}', refresh=True)
        # zero out epoch based performance variables
        all_acc = {key: 0 for key in phases}
        # keep this on GPU for greatly improved performance
        all_loss = {
            key: torch.zeros(0, dtype=torch.float64).to(device)
            for key in phases
        }
        cmatrix = {key: np.zeros((num_classes, num_classes)) for key in phases}

        my_model = MyModel(model, criterion, device, all_acc, all_loss,
                           cmatrix)

        print(
            "########################   STARTING TRAINING!  ########################\n"
        )

        my_model.train_model(optim, dataLoader['train'])

        print(
            "########################   STARTING EVALUATION!  ########################\n"
        )
        my_model.eval(dataLoader['val'])

        all_acc, all_loss, cmatrix = my_model.all_acc, my_model.all_loss, my_model.cmatrix

        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        if all_loss["val"] < best_loss_on_test:

            print(
                "########################   SAVING BEST MODEL!  ########################\n"
            )
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

            manager.save_model(model_name=f"{dataname}_densenet_best_model",
                               model=model,
                               model_info=state,
                               overwrite_model=True)
            # torch.save(state, f"output/{dataname}_densenet_best_model.pth")

    diagnose_example(model, manager, dataset, device, labels)


def diagnose_example(model, manager, dataset, device, labels):
    """Diagnose example"""

    print(
        "########################   DIAGNOSING EXAMPLE!  ########################\n"
    )
    img, label = dataset["val"][2]

    print(
        "########################   LOADING MODEL!  ########################\n"
    )
    checkpoint = manager.load_model(f"{dataname}_densenet_best_model")
    # checkpoint = torch.load(f"output/{dataname}_densenet_best_model.pth")

    print(
        "########################   LOADING STATE DICT!  ########################\n"
    )
    # model.load_state_dict(checkpoint.state_dict()['model_dict'])
    # model.load_state_dict(checkpoint["model_dict"])

    print(
        "########################   GENERATING OUTPUT!  ########################\n"
    )
    output = model(torch.Tensor(img[None, ::]).permute(0, 3, 1,
                                                       2).float()).to(device)
    output = output.detach().squeeze().cpu().numpy()
    label_decoder = lambda x: list(labels.keys())[list(labels.values()).index(
        x)]
    print(f"True class: {label_decoder(label)}\n")
    print(f"Predicted class: {label_decoder(np.argmax(output))}")


if __name__ == "__main__":
    main()
