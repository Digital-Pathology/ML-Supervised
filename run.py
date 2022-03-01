from types import new_class
from dataset import Dataset
from filtration import FilterManager, FilterBlackAndWhite, FilterHSV
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
parser.add_argument('--labels', type=str, default='labels_initial.csv')

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
labels = args['labels']
phases = ["train"]  # how many phases did we create databases for?
# when should we do valiation? note that validation is *very* time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch
validation_phases = []
# additionally, using simply [], will skip validation entirely, drastically speeding things up


def main():
    train_dir = SM_CHANNEL_TRAIN
    # test_dir = SM_CHANNEL_TEST
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
    for phase in phases:  # now for each of the phases, we're creating the dataloader
        # interestingly, given the batch size, i've not seen any improvements from using a num_workers>0
        dataset[phase] = Dataset(
            data_dir=train_dir, labels=labels, filtration=filtration)
        dataLoader[phase] = DataLoader(dataset[phase], batch_size=batch_size,
                                       shuffle=True, num_workers=0, pin_memory=True)
        print(f"{phase} dataset size:\t{len(dataset[phase])}")
    criterion = nn.CrossEntropyLoss()

    best_loss_on_test = np.Infinity
    edge_weight = 1.0
    edge_weight = torch.tensor(edge_weight).to(device)
    for epoch in tqdm(range(num_epochs)):
        # zero out epoch based performance variables
        all_acc = {key: 0 for key in phases}
        # keep this on GPU for greatly improved performance
        all_loss = {key: torch.zeros(0).to(device) for key in phases}
        cmatrix = {key: np.zeros((num_classes, num_classes)) for key in phases}

        for phase in phases:  # iterate through both training and validation states

            if phase == 'train':
                model.train()  # Set model to training mode
            else:  # when in eval mode, we don't want parameters to be updated
                model.eval()   # Set model to evaluate mode

            # for each of the batches
            for ii, (X, label) in enumerate(dataLoader[phase]):
                X = X.to(device)  # [Nbatch, 3, H, W]
                # [Nbatch, 1] with class indices (0, 1, 2,...num_classes)
                label = torch.tensor(list(map(lambda x: int(x), label))).to(device)

                # dynamically set gradient computation, in case of validation, this isn't needed
                with torch.set_grad_enabled(phase == 'train'):
                    # disabling is good practice and improves inference time

                    prediction = model(X.permute(0, 3, 1, 2).float())  # [N, Nclass]
                    loss = criterion(prediction, label)

                    if phase == "train":  # in case we're in train mode, need to do back propogation
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        train_loss = loss

                    all_loss[phase] = torch.cat(
                        (all_loss[phase], loss.detach().view(1, -1)))

                    if phase in validation_phases:  # if this phase is part of validation, compute confusion matrix
                        p = prediction.detach().cpu().numpy()
                        cpredflat = np.argmax(p, axis=1).flatten()
                        yflat = label.cpu().numpy().flatten()

                        cmatrix[phase] = cmatrix[phase] + \
                            confusion_matrix(yflat, cpredflat,
                                             labels=range(num_classes))

            all_acc[phase] = (cmatrix[phase]/cmatrix[phase].sum()).trace()
            all_loss[phase] = all_loss[phase].cpu().numpy().mean()


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

            torch.save(state, f"{dataname}_densenet_best_model.pth")
        else:
            print("")
    img, label, _ = dataset["val"][2]

    checkpoint = torch.load(f"{dataname}_densenet_best_model.pth")
    model.load_state_dict(checkpoint["model_dict"])
    output = model(img[None, ::].to(device))
    output = output.detach().squeeze().cpu().numpy()
    print(f"True class:{label}")
    print(f"Predicted class:{np.argmax(output)}")


if __name__ == "__main__":
    main()
