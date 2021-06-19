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
from typing import List, Tuple, Dict

from .Utils import log_number_of_classes, argnearest, load_training_mask
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


def N_out_conv(n_in, layer, dim=0):
    return int((n_in + 2 * layer.padding[dim] - layer.dilation[dim] * (layer.kernel_size[dim] - 1) - 1) / layer.stride[dim] + 1)


def N_out_pool(n_in, layer, dim=0):
    return int((n_in + 2 * layer.padding - layer.kernel_size[dim]) / layer.stride[dim] + 1)


class Conv2DUnit(nn.Module):
    def __init__(self, in_shape, nfltrs, kernel, stride, padding, resnet=False):
        super(Conv2DUnit, self).__init__()

        # 1. layer
        self.conv1 = nn.Conv2d(in_shape, in_shape, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_shape)

        # 2. layer
        self.conv2 = nn.Conv2d(
            in_shape, in_shape,
            kernel_size=tuple(kernel),
            padding=tuple(padding),
            stride=tuple(stride),
            padding_mode='circular',
            bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=in_shape)

        # 3. layer
        self.conv3 = nn.Conv2d(in_shape, nfltrs, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=nfltrs)

        self.relu = nn.ELU()

        #if self.pool is not None:
        #    self.pool = nn.AvgPool2d(tuple(pool))

        self.resnet = resnet
        if resnet:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_shape, nfltrs, kernel_size=(1, 1), padding=(0, 0), stride=tuple(stride), bias=False),
                nn.BatchNorm2d(num_features=nfltrs),
                #nn.AvgPool2d(tuple(pool))
            )

    def forward(self, input):
        # 1. layer
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)

        # 2. layer
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        # 3. layer
        output = self.conv3(output)
        output = self.bn3(output)
        #output = self.pool(output)

        if self.resnet:
            # skip connection
            ident = self.downsample(input)
            output += ident

        output = self.relu(output)


#        n_H = N_out_conv(input.shape[-2], self.conv, dim=0)
#        n_W = N_out_conv(input.shape[-1], self.conv, dim=1)
#
#        n_H = N_out_pool(n_H, self.pool, dim=0)
#        n_W = N_out_pool(n_W, self.pool, dim=1)
#
#        print(f'calc H/W out = {n_H} x {n_W}')
#        print(output.shape, ident.shape)
#
#        n_H = N_out_conv(input.shape[-2], self.downsample, dim=0)
#        n_W = N_out_conv(input.shape[-1], self.downsample, dim=1)
#
#        n_H = N_out_pool(n_H, self.pool, dim=0)
#        n_W = N_out_pool(n_W, self.pool, dim=1)
#
#        print(f'calc skip H/W out = {n_H} x {n_W}')

        return output


class DenseUnit(nn.Module):
    def __init__(self, in_shape, nnodes, dropout=0.0):
        super(DenseUnit, self).__init__()

        self.dense = nn.Linear(in_shape, nnodes)
        self.bn = nn.BatchNorm1d(num_features=nnodes)
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):

        output = self.dense(input)
        output = self.bn(output)
        output = self.relu(output)
        output = self.dropout(output)

        return output


class VoodooNet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int],
            output_shape: int,
            dense_layers: Tuple[int] = None,
            num_filters: Tuple[int] = None,
            kernel_sizes: Tuple[Tuple[int]] = None,
            stride_sizes: Tuple[Tuple[int]] = None,
            pool_sizes: Tuple[Tuple[int]] = None,
            pad_sizes: Tuple[Tuple[int]] = None,
            learning_rate: float = 1.0e-5,
            lr_decay: float = None,
            momentum: float = None,
            optimizer: str = None,
            dev: str = None,
            loss: str = None,
            resnet: bool = False,
            kernel_init: str = None,
            regularizer: str = None,
            batch_norm: str = True,
            hidden_activations: str = 'elu',
            output_activation: str = None,
            dropout: float = 0.0,
            **kwargs: dict):
        """
        Defining a PyTorch model.

        Args:
            input_shape (tuple): shape of the input tensor
            output_shape (tuple): shape of the output tensor

        Keyword Args:
            dense_layers: number of dense layers
            num_filters: list containing the number of nodes per conv layer
            kernel_sizes: list containing the 2D kernel
            stride_sizes: dimensions of the stride
            pool_sizes: dimensions of the pooling layers
            kernel_init: weight initializer method
            regularizer: regularization strategy
            hidden_activations: name of the activation functions for the convolutional layers
            output_activation: name of the activation functions for the dense output layer
            batch_norm: normalize the input layer by adjusting and scaling the activations
            dropout: percentage/100 of randome neuron dropouts during training
            metrics: list of metrics used for training

        """

        super(VoodooNet, self).__init__()
        self.n_conv = len(num_filters) if num_filters is not None else 0
        self.n_dense = len(dense_layers) if dense_layers is not None else 0
        self.batch_norm = batch_norm
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pool_sizes = np.array(pool_sizes)
        self.stride_sizes = np.array(stride_sizes)
        self.kernel_sizes = np.array(kernel_sizes)
        if pad_sizes is not None:
            self.pad_sizes = np.array(pad_sizes)
        else:
            self.pad_sizes = np.array([[int(kern[0]/2), int(kern[1]/2)] for kern in kernel_sizes])

        self.num_filters = np.array(num_filters)
        self.dense_layers = np.array(dense_layers)
        self.dropout = dropout
        self.device = self.get_device(dev if dev is not None else 'cuda:0')
        self.training = True
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.output_activation = nn.Softmax if output_activation else identity
        self.loss = self.get_loss_function(loss)
        self.lambda1 = lambda x: 0.95
        self.optimizer = self.get_optimizer(optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR
        self.resnet = resnet

        # initialize convolutional layer
        self._define_cnn()

        # flatten last convolution
        self._define_flatten()

        # calculate flat dimension
        self.n_flat = self.flatten_conv()

        # initialize dense layers
        self._define_dense(dropout=dropout)

    def flatten_conv(self):
        x = torch.rand(((1,) + tuple(self.input_shape[1:])))
        x = self.convolution_network(x)
        x = self.flatten(x)
        return x.shape[1]

    def _define_cnn(self):

        in_shape = self.input_shape[1]
        iterator = enumerate(zip(self.num_filters, self.kernel_sizes, self.stride_sizes, self.pad_sizes))
        self.conv2D = OrderedDict()
        for i, (ifltrs, ikrn, istride, ipad) in iterator:
            # Create 14 layers of the unit with max pooling in between
            self.conv2D.update({f'conv2d_{i}': Conv2DUnit(in_shape, ifltrs, ikrn, istride, ipad, resnet=self.resnet)})
            in_shape = ifltrs

        # Add all the units into the Sequential layer in exact order
        self.convolution_network = nn.Sequential(self.conv2D)

    def _define_flatten(self):
        self.flatten = nn.Flatten()

    def _define_dense(self, dropout=0.0):

        log_TM.debug('calc flattened :', self.n_flat)
        in_shape = self.n_flat
        self.dense = OrderedDict()
        for i, inodes in enumerate(self.dense_layers):
            self.dense.update({f'dense_{i}': DenseUnit(in_shape, inodes, dropout=dropout)})
            in_shape = inodes

        # output layer
        self.dense.update({f'dense_{i + 1}': nn.Linear(in_shape, self.output_shape)})

        self.dense_network = nn.Sequential(self.dense)

    def save(self, path: str = None, aux: Dict = None):
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'auxiliaray': aux
        }
        torch.save(checkpoint, path)
        return 0

    def print_nparams(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        pytorch_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log_TM.info(f'Total non-trainable parameters: {pytorch_total_params - pytorch_trainable_params:,d}')
        log_TM.info(f'    Total trainable parameters: {pytorch_trainable_params:_d}')
        log_TM.critical(f'             Total  parameters: {pytorch_total_params:_d}')

    def forward(self, x):
        x = self.convolution_network(x)
        x = self.flatten(x)
        x = self.dense_network(x)
        return x

    def fwd_pass(self, X, y, train=False):
        if train:
            self.zero_grad()

        outputs = self(X)
        metrics = evaluation_metrics(outputs[:, 1] > 0.5, y)
        loss = self.loss(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return metrics, loss

    def randome_test(self, X_test, y_test, test_size=256, dev='cpu'):
        random_start = np.random.randint(len(X_test) - test_size)
        if test_size > 0:
            X = X_test[random_start:random_start + test_size]
            y = y_test[random_start:random_start + test_size]

            with torch.no_grad():
                val_metrics, val_loss = self.fwd_pass(X.to(dev), y.to(dev))
        else:
            val_metrics, val_loss = [], []
            iterator = tqdm(
                range(0, len(X_test), 256),
                ncols=100,
                unit=f' batches - validation'
            )
            for i in iterator:
                X = X_test[i:i + 256].to(dev)
                y = y_test[i:i + 256].to(dev)

                with torch.no_grad():
                    _1, _2 = self.fwd_pass(X.to(dev), y.to(dev))
                    val_metrics.append(_1['array'])
                    val_loss.append(_2.numpy())

            val_metrics = np.mean(val_metrics, axis=0)
            val_loss = np.mean(val_loss)

        return val_metrics, val_loss

    def optimize(self, X, y, X_test, y_test, batch_size=100, epochs=10, dev='cpu'):
        self.to(dev)
        self.train()
        statistics = []
        log_TM.info('\nOptimize')

        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)
        self.lr_scheduler = self.lr_scheduler(self.optimizer, step_size=50, gamma=0.1)

        for epoch in range(epochs):
            iterator = tqdm(
                range(0, len(X), batch_size),
                ncols=100,
                unit=f' batches - epoch:{epoch + 1}/{epochs}'
            )
            for i in iterator:
                # show_batch(X[i:i+batch_size])
                batch_X = X[i:i + batch_size].to(dev)
                batch_y = y[i:i + batch_size].to(dev)
                if len(batch_y) < 2: continue

                batch_metric, batch_loss = self.fwd_pass(batch_X, batch_y, train=True)

                if i % 5 == 0:
                    self.lr_scheduler.step()
                    val_metric, val_loss = self.randome_test(
                        X_test, y_test, test_size=1024, dev=dev
                    )
                    statistics.append(
                        [np.append(batch_metric, batch_loss.to('cpu').detach().numpy()),
                         np.append(val_metric, val_loss.to('cpu').detach().numpy())]
                    )

        return statistics

    def testing(self, X_test, batch_size=2048, dev='cpu'):
        self.to(dev)
        self.eval()
        pred = []
        log_TM.info('\nTesting')
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test), batch_size), ncols=100, unit=' batches'):
                batch_X = X_test[i:i + batch_size].to(dev)
                pred.append(self(batch_X))

        activation = self.output_activation(dim=1)
        return activation(torch.cat(pred, 0))


    # static methods

    @staticmethod
    def get_device(dev):
        if torch.cuda.is_available():
            log_TM.info("Available GPUs:", torch.cuda.device_count())
            device = torch.device(dev)  # you can continue going on here, likrne cuda:1 cuda:2....etc.
        else:
            device = torch.device("cpu")
        log_TM.info(f"Running on {device}")
        return device

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
        elif string == 'adagrad':
            return torch.optim.Adagrad
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
    def fetch_data(
            path: str,
            shuffle: bool = True,
            balance: int = False,
            remove_classes: List[int] = None,
            garbage: List[int] = None,
            drop_pct: float = 0.0,
            merge_classes: Dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        t0 = time.time()
        with xr.open_zarr(path) as data:
            X = data['features'].values
            y = data['targets'].values

            print(f'time elapsed reading {path} :: {int(time.time() - t0)} sec')

            assert len(X) != 4, f'Input data has wrong shape: {X.shape}'

            ichannel = 0
            X = X[:, :, :, ichannel]  # good!
            X = X[:, :, :, np.newaxis]

            # nsamples, npolarization, ntimesteps, nDopplerbins
            X = X.transpose(0, 3, 2, 1)
            log_number_of_classes(y, text=f'\nsamples per class in {path}')

            if garbage is not None:
                for i in garbage:
                    y[y == i] = 10

            if remove_classes is not None:
                for iclass in remove_classes:
                    X, y = VoodooNet.remove_randomely(100, iclass, X, y)
                log_number_of_classes(y, text=f'\nsamples per class, classes removed: {*remove_classes,}')

            if merge_classes is not None:
                tmp = y.copy()
                for key, val in merge_classes.items():  # i from 0, ..., ngroups-1
                    for jclass in val:
                        tmp[y == jclass] = key
                y = tmp

            if balance > 0:
                for i in range(11):
                    X, y = VoodooNet.remove_randomely(balance, i, X, y)
                log_number_of_classes(y, text=f'\nsamples per class balanced')

            X = torch.Tensor(X)
            y = torch.Tensor(y)
            y = y.type(torch.LongTensor)

            if shuffle:
                perm = torch.randperm(len(y))
                X, y = X[perm], y[perm]

            if 0 < drop_pct < 1:
                idx_drop = int(X.shape[0] * drop_pct)
                X, y = X[idx_drop:, ...], y[idx_drop:]

            return X, y

    @staticmethod
    def fetch_2Ddata(path):
        # load 2d data
        with xr.open_zarr(path) as data:
            ds = data.copy()
            ds['classes'].attrs = {'colormap': 'cloudnet_target_new', 'rg_unit': 'km', 'var_unit': '', 'system': 'Cloudnetpy', 'var_lims': [0, 10]}
            ds['status'].attrs = {'colormap': 'cloudnetpy_detection_status', 'rg_unit': 'km', 'var_unit': '', 'system': 'Cloudnetpy', 'var_lims': [0, 7]}
            ds['Ze'].attrs = {'colormap': 'jet', 'rg_unit': 'km', 'var_unit': 'dBZ', 'system': 'Cloudnetpy', 'var_lims': [-50, 20]}
        return ds

    @staticmethod
    def remove_randomely(remove_from_class, class_nr, _feature_set, _target_labels):
        idx = np.where(_target_labels == class_nr)[0]
        perm = np.arange(0, idx.size)
        np.random.shuffle(perm)
        idx = idx[perm]

        if 0.0 <= remove_from_class <= 100.0:
            rand_choice = idx[:int(idx.size * remove_from_class / 100.)]
        elif remove_from_class > 100:
            idx_cut = remove_from_class if len(idx) > remove_from_class else -2
            rand_choice = idx[idx_cut:]
        else:
            raise ValueError(f'Choose value above 0, not {remove_from_class}')

        _feature_out = np.delete(_feature_set, rand_choice, axis=0)
        _target_out = np.delete(_target_labels, rand_choice, axis=0)
        return _feature_out, _target_out

    @staticmethod
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

    @staticmethod
    def reshape(pred, mask, newshape):

        pred_reshaped = np.zeros(newshape)
        cnt = 0
        for i in range(newshape[0]):
            for j in range(newshape[1]):
                if mask[i, j]: continue
                pred_reshaped[i, j] = pred[cnt]
                cnt += 1

        return pred_reshaped

    @staticmethod
    def make_weights_for_balanced_classes(images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight

    @staticmethod
    def smooth_prediction(probabilities, shape):
        # plot smoothed data
        from scipy.ndimage import gaussian_filter

        smoothed_probs = np.zeros(shape)
        for i in range(shape[2]):
            smoothed_probs[:, :, i] = gaussian_filter(probabilities[:, :, i], sigma=1)

        return smoothed_probs

    @staticmethod
    def random_subset(xr_ds, var='CLASS', rg_max=12000):

        # choose a random set of indices
        analyse_classes = [2, 4]
        Nclasses = log_number_of_classes(
            xr_ds['CLASS'].values,
            text='\nsamples per class, from prediction'
        )

        analyse_classes = [i for i in analyse_classes if Nclasses[i] > len(analyse_classes)]
        N = len(analyse_classes)

        indices = np.concatenate(
            [random_choice(
                xr_ds,
                rg_max,
                N=N,
                iclass=i,
                var=var) for i in analyse_classes],
            axis=0)
        return indices

    @staticmethod
    def to_nc(XnD, X2D, prediction, path='', n_class=11, **kwargs):
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
            VoodooNet.new_classification(
                prediction, xr_ds['mask'].values
            ),
            **kwargs
        )
        xr_ds['CLOUDNET_CLASS'] = X2D['classes'].copy()
        xr_ds['CLOUDNET_STATUS'] = X2D['status'].copy()
        xr_ds['CLOUDNET_ZE'] = X2D['Ze'].copy()

        xr_ds.add_nD_variable(
            'CLOUDNET_STATUS', ('ts', 'rg'),
            VoodooNet.new_classification(
                prediction,
                xr_ds['mask'].values
            ),
            **kwargs
        )

        xr_ds.add_nD_variable(
            'PROBDIST', ('ts', 'rg', 'cl'),
            VoodooNet.reshape(
                prediction,
                xr_ds['mask'].values,
                (n_ts, n_rg, n_class)
            ),
            **kwargs
        )

        xr_ds.add_nD_variable(
            'ZSpec', ('ts', 'rg', 'ch', 'vel'),
            VoodooNet.reshape(
                XnD[:, POLARIZ, :, :],
                xr_ds['mask'].values,
                (n_ts, n_rg, n_ch, n_vel)
            ),
            **kwargs
        )

        sprobabilities = VoodooNet.smooth_prediction(
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

def identity(input):
    return input


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


def evaluation_metrics(pred_labels, true_labels, status=None):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # if no status is given, metric is calculated during optimization
    if status is None:
        for pred, truth in zip(pred_labels, true_labels):
            if truth in [1, 3, 5, 7]:  # TRUE
                if pred == 1:  # is droplet
                    TP += 1
                else:
                    FN += 1
            else:  # FALSE
                if pred == 1:
                    FP += 1
                else:
                    TN += 1


    # if status is given, prediction and true labels in time-range
    else:
        evaluation_mask = ~load_training_mask(true_labels, status)
        # also remove good radar echos only
        evaluation_mask[status == 2] = False

        n_time, n_range = true_labels.shape
        for ind_time in range(n_time):
            for ind_range in range(n_range):
                if evaluation_mask[ind_time, ind_range]:
                    cloundet_label = true_labels[ind_time, ind_range]
                    predicted_label = pred_labels[ind_time, ind_range]
                    if cloundet_label in [1, 3, 5, 7]:  # TRUE
                        if predicted_label == 1:  # is droplet
                            TP += 1
                        else:
                            FN += 1
                    else:  # FALSE
                        if predicted_label == 1:
                            FP += 1
                        else:
                            TN += 1

    from collections import OrderedDict
    out = OrderedDict({
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'precision': TP / max(TP + FP, 1.0e-7),
        'npv': TN / max(TN + FN, 1.0e-7),
        'recall': TP / max(TP + FN, 1.0e-7),
        'specificity': TN / max(TN + FP, 1.0e-7),
        'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
        'F1-score': 2 * TP / max(2 * TP + FP + FN, 1.0e-7),
        'Jaccard-index': TP / max(TP + FN + FP, 1.0e-7),
    })
    out.update({
        'array': np.array([val for val in out.values()], dtype=float)
    })
    return out
