"""
This module contains functions for generating deep learning models with Tensorflow and Keras.

"""

import os
import sys
import logging
import time
from collections import OrderedDict

import matplotlib.pyplot as plt

import numpy as np

np.random.seed(0)

import torch
import torch.nn as nn

torch.manual_seed(0)

import xarray as xr

from tqdm.auto import tqdm, trange
from typing import List, Tuple

from .Utils import log_number_of_classes, argnearest
from .Utils import logger as log_UT
from .Loader import preproc_ini, VoodooXR

log_TM = logging.getLogger(__name__)
log_TM.setLevel(logging.CRITICAL)
log_UT.setLevel(logging.INFO)

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2021, The Voodoo Project"
__credits__ = ["Willi Schimmel"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

def random_choice(xr_ds, rg_int, N=4, iclass=4, var='CLASS'):
    nts, nrg = xr_ds.ZSpec.ts.size, xr_ds.ZSpec.rg.size

    icnt = 0
    indices = np.zeros((N, 2), dtype=np.int)
    nnearest = argnearest(xr_ds.ZSpec.rg.values, rg_int)
    # mask_below_x = xr_ds['PROBDIST'][:, :, iclass].values > 0.5

    MAXITER = 1000
    while icnt < N:
        iter = 0
        while True:
            iter += 1
            idxts = int(np.random.randint(0, high=nts, size=1))
            idxrg = int(np.random.randint(0, high=nnearest, size=1))
            msk = ~xr_ds.mask[idxts, idxrg]  # * mask_below_x[idxts, idxrg]
            cls = xr_ds[var].values[idxts, idxrg] == iclass
            # if iter > MAXITER:
            #    raise RuntimeError(f'No class {iclass} found!')
            if msk and cls:
                indices[icnt, :] = [idxts, idxrg]
                icnt += 1
                break
    return indices

def smooth_prediction(probabilities, shape):
    # plot smoothed data
    from scipy.ndimage import gaussian_filter

    smoothed_probs = np.zeros(shape)
    for i in range(shape[2]):
        smoothed_probs[:, :, i] = gaussian_filter(probabilities[:, :, i], sigma=1)

    return smoothed_probs

def new_classification(pred, mask):

    nts, nrg = mask.shape
    classes = np.zeros(mask.shape)
    cnt = 0
    pred_classes = torch.argmax(pred, 1)

    for i in range(nts):
        for j in range(nrg):
            if mask[i, j]: continue
            classes[i, j] = pred_classes[cnt]
            cnt += 1

    return classes

def new_classification_post(pred, mask):

    nts, nrg = mask.shape
    classes = np.zeros(mask.shape)
    cnt = 0

    pred_classes = torch.argmax(pred, 1)

    for i in range(nts):
        for j in range(nrg):
            if mask[i, j]: continue
            classes[i, j] = pred_classes[cnt]
            cnt += 1

    return classes

def reshape(pred, mask, newshape):

    pred_reshaped = np.zeros(newshape)
    cnt = 0
    for i in range(newshape[0]):
        for j in range(newshape[1]):
            if mask[i, j]: continue
            pred_reshaped[i, j] = pred[cnt]
            cnt += 1

    return pred_reshaped
############


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()

        ## willi

        self.optimizer = self.get_optimizer('adam')
        self.loss = self.get_loss_function('crossentropy')

        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, intermediate_channels * self.expansion, kernel_size=1, stride=stride),
            nn.BatchNorm2d(intermediate_channels * self.expansion)
        )
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion  # 256

        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels))  # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)

    def fwd_pass(self, X, y, train=False):
        if train:
            self.zero_grad()

        outputs = self(X)
        matches = [torch.argmax(i) == j for i, j in zip(outputs, y)]
        acc = matches.count(True) / len(matches)
        loss = self.loss(outputs, y)

        if train:
            optimizer = self.optimizer(self.parameters(), lr=1.0e-3)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            optimizer.step()

        return acc, loss


    def optimize(self, X, y, X_test, y_test, batch_size=100, epochs=10, dev='cpu'):
        self.to(dev)
        self.train()
        stat = []
        log_TM.info('\nOptimize')
        batch_acc, batch_loss = 0, 0
        for epoch in range(epochs):
            epA, epL = [], []
            epVA, epVL = [], []
            iterator = tqdm(
                range(0, len(X), batch_size),
                ncols=100,
                unit=f' batches - epoch:{epoch + 1}/{epochs}'
            )
            for i in iterator:
                # show_batch(X[i:i+batch_size])
                batch_X = X[i:i + batch_size].to(dev)
                batch_y = y[i:i + batch_size].to(dev)

                batch_acc, batch_loss = self.fwd_pass(batch_X, batch_y, train=True)

                if i % 50 == 0:
                    val_acc, val_loss = self.randome_test(X_test, y_test, dev=dev)
                    epA.append(batch_acc)
                    epVA.append(val_acc)
                    epL.append(float(batch_loss))
                    epVL.append(float(val_loss))
                    stat.append([i, np.mean(epA), np.mean(epVA), np.mean(epL), np.mean(epVL)])
            log_TM.info(f'ep{epoch + 1:4d}/{epochs} Acc: {stat[-1][1]:.2f} / {stat[-1][2]:.2f}  '
                        f'Loss: {stat[-1][3]:.8f} / {stat[-1][4]:.8f}')

        return np.array(stat)

    def testing(self, X_test, batch_size=2048, dev='cpu'):
        self.to(dev)
        self.eval()
        pred = []
        log_TM.info('\nTesting')
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test), batch_size), ncols=100, unit=' batches'):
                batch_X = X_test[i:i + batch_size].to(dev)
                pred.append(self(batch_X))

        sm = nn.Softmax(dim=1)
        return sm(torch.cat(pred, 0))

    @staticmethod
    def get_optimizer(string):
        if string == 'sgd':
            return torch.optim.SGD
        elif string == 'Nadam':
            return torch.optim.Nadam
        elif string == 'rmsprop':
            return torch.optim.RMSprop
        elif string == 'adam':
            return torch.optim.Adam
        else:
            raise ValueError('Unknown OPTIMIZER!', string)

    @staticmethod
    def get_loss_function(string):
        if string == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError('Unknown loss function!', string)

    @staticmethod
    def load(model_path: str = None):
        if os.path.exists(model_path):
            log_TM.info(f'Loaded model from disk:\n{model_path}')
            return torch.load(model_path)
        else:
            raise RuntimeError(f'No model found for path: {model_path}')

    @staticmethod
    def to_nc(XnD, X2D, prediction, path='', n_class=3, **kwargs):
        # only for one poliarization and fixed 11 clases
        # create xr for new prediction
        POLARIZ = 0
        (n_ts, n_rg), (n_ch, n_vel) = X2D['mask'].shape, XnD.shape[2:]
        xr_ds = VoodooXR(X2D['ts'].values, X2D['rg'].values)
        xr_ds.add_coordinate({'cl': np.arange(0, n_class)}, 'Number of classes')
        xr_ds.add_coordinate({'vel': np.arange(0, n_vel)}, 'Number of spectral bins')
        xr_ds.add_coordinate({'ch': np.arange(0, n_ch)}, 'Number of channels')

        xr_ds['mask'] = X2D['mask'].copy()

        xr_ds.add_nD_variable(
            'CLASS', ('ts', 'rg'),
            new_classification(
                prediction, xr_ds['mask'].values
            ),
            **kwargs
        )
        xr_ds['CLOUDNET_CLASS'] = X2D['classes'].copy()
        xr_ds['CLOUDNET_STATUS'] = X2D['status'].copy()
        xr_ds['CLOUDNET_ZE'] = X2D['Ze'].copy()

        xr_ds.add_nD_variable(
            'CLOUDNET_STATUS', ('ts', 'rg'),
            new_classification(
                prediction,
                xr_ds['mask'].values
            ),
            **kwargs
        )

        xr_ds.add_nD_variable(
            'PROBDIST', ('ts', 'rg', 'cl'),
            reshape(
                prediction,
                xr_ds['mask'].values,
                (n_ts, n_rg, n_class)
            ),
            **kwargs
        )

        xr_ds.add_nD_variable(
            'ZSpec', ('ts', 'rg', 'ch', 'vel'),
            reshape(
                XnD[:, POLARIZ, :, :],
                xr_ds['mask'].values,
                (n_ts, n_rg, n_ch, n_vel)
            ),
            **kwargs
        )

        sprobabilities = smooth_prediction(
            xr_ds['PROBDIST'].values,
            (n_ts, n_rg, n_class)
        )
        xr_ds.add_nD_variable(
            'sCLASS', ('ts', 'rg'),
            np.argmax(sprobabilities, axis=2),
            **kwargs
        )
        xr_ds['sCLASS'].values[xr_ds['mask'].values] = 0
        xr_ds['sCLASS'] = xr_ds['sCLASS'].astype(int)
        xr_ds['CLASS'] = xr_ds['CLASS'].astype(int)

        if len(path) > 0:
            xr_ds.to_netcdf(path, mode='w')
            log_TM.critical(path)

        return xr_ds

    def randome_test(self, X_test, y_test, test_size=100, dev='cpu'):
        random_start = np.random.randint(len(X_test) - test_size)
        X = X_test[random_start:random_start + test_size]
        y = y_test[random_start:random_start + test_size]
        with torch.no_grad():
            val_acc, val_loss = self.fwd_pass(X.to(dev), y.to(dev))
        return val_acc, val_loss

    def save(self, path: str = None):
        torch.save(self, path)
        return 0

    def print_nparams(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        pytorch_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log_TM.info(f'Total non-trainable parameters: {pytorch_total_params - pytorch_trainable_params:,d}')
        log_TM.info(f'    Total trainable parameters: {pytorch_trainable_params:_d}')
        log_TM.critical(f'             Total  parameters: {pytorch_total_params:_d}')




def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(18, Block, img_channels, num_classes)


def ResNet34(img_channels=3, num_classes=1000):
    return ResNet(34, Block, img_channels, num_classes)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(50, Block, img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(101, Block, img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(152, Block, img_channels, num_classes)


def test():
    net = ResNet18(img_channels=1, num_classes=11)
    y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    print(y.size())


#test()
