import torch
import torch.nn as nn
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from torch import Tensor
from torch import functional as F

import torchvision
import torchvision.transforms as transforms

from torchmetrics import Accuracy

from model.vig import PyramidViG
from typing import Tuple, List

from pytorch_lightning import Trainer


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='cifar100',
                        help='dataset used to perform the test (options: cifar10 or cifar100)')
    parser.add_argument('-rpe', action='store_true',
                        help='usage of relative positional embedding between patches(nodes) of the images')
    parser.add_argument('-gl', type=str, default='default',
                        help='grapher layer (options: default, GAT, GCN)')

    # parameters for comet logger
    parser.add_argument('--project-name', type=str, dest='project_name',
                        default='Master thesis - extended tests')
    parser.add_argument('--workspace', type=str, default='giovasciortino')
    parser.add_argument('--expname', type=str,
                        default='test', help='version of the test')
    return parser.parse_args()


class PyramidViGLT(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 heads: int,
                 n_classes: int,
                 input_resolution: Tuple[int, int],
                 reduce_factor: int,
                 pyramid_reduction: int = 2,
                 act: str = 'relu',
                 k: int = 4,
                 overlapped_patch_emb: bool = True,
                 relative_positional_embedding: bool = True,
                 grapher_layer: str = 'default',
                 dataset: str = None,
                 batch_size: int = -1,
                 **kwargs) -> None:
        super(PyramidViGLT, self).__init__()
        self.model = PyramidViG(in_channels,
                                out_channels,
                                heads,
                                n_classes,
                                input_resolution,
                                reduce_factor,
                                pyramid_reduction,
                                act,
                                k,
                                overlapped_patch_emb,
                                relative_positional_embedding,
                                grapher_layer,
                                **kwargs)
        self.batch_size = batch_size
        self.dataset = dataset
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        return

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch):
        x, y = train_batch
        out = self.model(x)
        out = out.squeeze(-1).squeeze(-1)

        y = y.type(torch.LongTensor).to(device)

        loss = self.loss(out, y)
        out = torch.argmax(out, dim=1)

        self.acc.update(out, y)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        acc = self.acc.compute()
        self.log("training_epoch_average", acc)

        self.acc.reset()
        return

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.log("validation_epoch_average", acc)
        self.val_acc.reset()
        return

    def validation_step(self, val_batch):
        x, y = val_batch
        out = self.model(x)
        out = out.squeeze(-1).squeeze(-1)
        y = y.type(torch.LongTensor).to(device)

        loss = self.loss(out, y)
        out = torch.argmax(out, dim=1)

        self.val_acc.update(out, y)
        self.log("val_loss", loss)
        return

    def backward(self, loss: Tensor) -> None:
        loss.backward()
        return

    def train_dataloader(self):
        if self.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=True, transform=transforms.ToTensor(), download=True)
        else:
            train_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=True, transform=transforms.ToTensor(), download=True)
        # Data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )
        return train_loader

    def val_dataloader(self):

        if self.dataset == 'cifar10':
            test_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=False, transform=transforms.ToTensor(), download=True)
        else:
            test_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=False, transform=transforms.ToTensor(), download=True)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )
        return test_loader


if __name__ == '__main__':
    # INPUT PROGRAM ARGUMENTS
    args = parse()
    DATASET = args.d
    RELATIVE_POSITIONAL_EMBEDDING = args.rpe
    GRAPH_LAYER = args.gl
    # parameters for comet logger
    PROJECT_NAME = args.project_name
    WORKSPACE = args.workspace
    EXPERIMENT_NAME = args.expname
    # END INPUT PROGRAM ARGUMENTS

    assert GRAPH_LAYER in {'default', 'GAT',
                           'GCN'}, f'GRAPH_LAYER invalid ({GRAPH_LAYER})'
    assert DATASET in {'cifar10', 'cifar100'}, f'DATASET invalid ({DATASET})'

    # Hyper-parameters
    in_channels = 3
    input_resolution = (32, 32) #input resolution for cifar dataset
    heads = 2
    n_classes = 10 if DATASET == 'cifar10' else 100

    out_channels = [32, 64, 128]
    num_epochs = 300
    batch_size = 32
    reduce_factor = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(api_key=os.environ["COMET_API_KEY"],
                               project_name=PROJECT_NAME,
                               workspace=WORKSPACE,
                               experiment_name=EXPERIMENT_NAME)

    model = PyramidViGLT(in_channels, out_channels, heads,  n_classes, input_resolution, reduce_factor,
                         relative_positional_embedding=RELATIVE_POSITIONAL_EMBEDDING, grapher_layer=GRAPH_LAYER, dataset=DATASET, batch_size=batch_size)

    trainer = Trainer(max_epochs=num_epochs, devices=1,
                      logger=comet_logger, check_val_every_n_epoch=10)
    trainer.fit(model)
