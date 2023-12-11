import comet_ml
import torch
import torch.nn as nn
import argparse
import os
import torchgeo.datasets as datasets

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.lightning_wrapper import PyramidViGLT, Resnet101LT
from pytorch_lightning import Trainer

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=str, default='ben19',
                        help='dataset used to perform the test (options: ben19 or ben43)')
    parser.add_argument('--model', type=str, default='vig',
                        help='usage of relative positional embedding between patches(nodes) of the images')
    parser.add_argument('--rpe', action='store_true',
                        help='enable relative positional encoding')
    parser.add_argument('--gl', type=str, default='default',
                        help='grapher layer (options: default, GAT, GCN)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='batch size')
    parser.add_argument('--devices', type=int, default=1,
                        help='number of devuces')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to checkpoint')
    parser.add_argument('--rgb', action='store_true', help='rgb channels only')

    # parameters for comet logger
    parser.add_argument('--projectname', type=str, dest='project_name',
                        default='--')
    parser.add_argument('--workspace', type=str, default='--')
    parser.add_argument('--expname', type=str,
                        default='test', help='version of the test')
    return parser.parse_args()



if __name__ == '__main__':
    # INPUT PROGRAM ARGUMENTS
    args = parse()
    DATASET = args.d
    MODEL = args.model
    RELATIVE_POSITIONAL_EMBEDDING = args.rpe
    GRAPH_LAYER = args.gl
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    CKPT_PATH = args.ckpt
    DEVS = args.devices
    RGB = args.rgb
    # parameters for comet logger
    PROJECT_NAME = args.project_name
    WORKSPACE = args.workspace
    EXPERIMENT_NAME = args.expname
    # END INPUT PROGRAM ARGUMENTS

    assert GRAPH_LAYER in {'default', 'GAT',
                           'GCN'}, f'GRAPH_LAYER invalid ({GRAPH_LAYER})'
    assert DATASET in {'ben19', 'ben43'}, f'DATASET invalid ({DATASET})'
    assert MODEL in {'vig', 'resnet101'}, f'MODEL invalid ({MODEL})'

    # Hyper-parameters
    in_channels = 3 if RGB else 12
    # input resolution of bigearthnet sentinel 2 images
    input_resolution = (120, 120)
    heads = 2
    n_classes = 43

    print(f"Number of classes: {n_classes}\nRelative positional embedding: {RELATIVE_POSITIONAL_EMBEDDING}\nNumber of input channels: {in_channels}")

    out_channels = [128, 256, 512]

    num_epochs = EPOCHS
    batch_size = BATCH_SIZE
    reduce_factor = 2

    train_dataset = datasets.BigEarthNet(
            root="./data", split='train', bands='s2', num_classes=n_classes, transforms=nn.Identity(), download=True)
    val_dataset = datasets.BigEarthNet(
            root="./data", split='val', bands='s2', num_classes=n_classes, transforms=nn.Identity(), download=True)
    test_dataset = datasets.BigEarthNet(
            root="./data", split='test', bands='s2', num_classes=n_classes, transforms=nn.Identity(), download=True)
    
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(api_key=os.environ["COMET_API_KEY"],
                               project_name=PROJECT_NAME,
                               workspace=WORKSPACE,
                               experiment_name=EXPERIMENT_NAME)

    if CKPT_PATH is None:
        if MODEL== 'vig':
            model = PyramidViGLT(in_channels, out_channels, heads,  n_classes, input_resolution, reduce_factor,
                             relative_positional_embedding=RELATIVE_POSITIONAL_EMBEDDING, grapher_layer=GRAPH_LAYER, dataset=DATASET, batch_size=batch_size)
        else:
            model = Resnet101LT(in_channels=in_channels, num_classes = n_classes, batch_size = batch_size)
    else:
        if MODEL == 'vig':
            model = PyramidViGLT.load_from_checkpoint(CKPT_PATH, in_channels=in_channels, out_channels=out_channels, heads=heads,  n_classes=n_classes, input_resolution=input_resolution, reduce_factor=reduce_factor,
                                                  relative_positional_embedding=RELATIVE_POSITIONAL_EMBEDDING, grapher_layer=GRAPH_LAYER, dataset=DATASET, batch_size=batch_size)
        else:
            model = Resnet101LT.load_from_checkpoint(CKPT_PATH, in_channels=in_channels, num_classes = n_classes, batch_size = batch_size)

    default_root = './final_checkpoints/'

    callbacks = [EarlyStopping(monitor="loss/val", mode="min", patience=15),
                 ModelCheckpoint(dirpath=default_root+EXPERIMENT_NAME, filename='{epoch}', monitor="loss/val")]
    trainer = Trainer(max_epochs=num_epochs, devices=DEVS,
                      logger=comet_logger, check_val_every_n_epoch=2, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)