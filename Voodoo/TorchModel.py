"""
This module contains functions for generating deep learning models with Tensorflow and Keras.

"""

import logging
import os
import random
import time
from collections import OrderedDict
from typing import List, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm

from .Utils import logger as log_UT

log_TM = logging.getLogger(__name__)
log_TM.setLevel(logging.CRITICAL)
log_UT.setLevel(logging.INFO)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2021, The Voodoo Project"
__credits__ = ["Willi Schimmel"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


class Conv2DUnit(nn.Module):
    def __init__(self, in_shape, nfltrs, kernel, stride, padding):
        super(Conv2DUnit, self).__init__()

        self.conv = nn.Conv2d(
            in_shape, nfltrs,
            kernel_size=tuple(kernel),
            padding=tuple(padding),
            stride=tuple(stride),
            padding_mode='circular')

        self.bn = nn.BatchNorm2d(num_features=nfltrs)
        self.relu = nn.ELU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

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


class OutputUnit(nn.Module):
    def __init__(self, in_shape, nnodes, activation):
        super(OutputUnit, self).__init__()
        self.dense = nn.Linear(in_shape, nnodes)
        self.activation = activation(dim=1)

    def forward(self, input):
        output = self.dense(input)
        output = self.activation(output)
        return output


class VoodooNet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int],
            output_shape: int,
            **pts: dict):
        """
        Defining a PyTorch model.

        Args:
            input_shape: shape of the input tensor
            output_shape: shape of the output tensor
            pts: pytorch settings

        Keywargs pts:
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
        self.n_conv = len(pts['num_filters']) if pts['num_filters'] is not None else 0
        self.n_dense = len(pts['dense_layers']) if pts['dense_layers'] is not None else 0
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.stride_sizes = np.array(pts['stride_sizes'])
        self.kernel_sizes = np.array(pts['kernel_sizes'])
        if pts['pad_sizes'] is not None:
            self.pad_sizes = np.array(pts['pad_sizes'])
        else:
            self.pad_sizes = np.array([[kern[0] - 1, kern[1] - 2] for kern in pts['kernel_sizes']])

        self.num_filters = np.array(pts['num_filters'])
        self.dense_layers = np.array(pts['dense_layers'])
        self.dropout = pts['dropout']
        self.device = pts['dev']
        self.training = True
        self.lr = float(pts['learning_rate'])
        self.lr_decay = float(pts['learning_rate_decay'])
        self.lr_decay_step = float(pts['learning_rate_decay_step'])
        self.output_activation = nn.Softmax
        self.loss = self.get_loss_function(pts['loss'])
        self.optimizer = self.get_optimizer(pts['optimizer'])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR

        if pts['task'] == 'train':
            # Capture a dictionary of hyperparameters with config
            self.wandb = wandb.init(
                project=str(pts['Vnet_label']),
                name=str(pts['model_name']),
                entity="krljhnsn"
            )
            self.wandb.config.update(pts, allow_val_change=True)

        # initialize convolutional layer
        self.define_cnn()

        # flatten last convolution
        self.flatten = nn.Flatten()

        # calculate flat dimension
        self.n_flat = self.flatten_conv()

        # initialize dense layers
        self.define_dense(dropout=self.dropout)

    def flatten_conv(self):
        x = torch.rand(((1,) + tuple(self.input_shape[1:])))
        x = self.convolution_network(x)
        x = self.flatten(x)
        return x.shape[1]

    def define_cnn(self):

        in_shape = self.input_shape[1]
        iterator = enumerate(zip(self.num_filters, self.kernel_sizes, self.stride_sizes, self.pad_sizes))
        self.conv2D = OrderedDict()
        for i, (ifltrs, ikrn, istride, ipad) in iterator:
            # Create 14 layers of the unit with max pooling in between
            self.conv2D.update({f'conv2d_{i}': Conv2DUnit(in_shape, ifltrs, ikrn, istride, ipad)})
            in_shape = ifltrs

        # Add all the units into the Sequential layer in exact order
        self.convolution_network = nn.Sequential(self.conv2D)

    def define_dense(self, dropout=0.0):
        in_shape = self.n_flat
        self.dense = OrderedDict()
        for i, inodes in enumerate(self.dense_layers):
            self.dense.update({f'dense_{i}': DenseUnit(in_shape, inodes, dropout=dropout)})
            in_shape = inodes

        # output layer
        self.dense.update({f'dense_{len(self.dense_layers) + 1}': OutputUnit(in_shape, self.output_shape, self.output_activation)})
        self.dense_network = nn.Sequential(self.dense)

    def forward(self, x):
        x = self.convolution_network(x)
        x = self.flatten(x)
        x = self.dense_network(x)
        return x

    def fwd_pass(self, X, y, train=False):
        if train:
            self.zero_grad()

        outputs = self(X)
        metrics = evaluation_metrics(outputs[:, 0] > 0.5, y[:, 0] > 0.5)
        loss = self.loss(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return metrics, loss

    def train_log(self, values, example_ct, epoch):
        # Where the magic happens
        self.wandb.log(values, step=example_ct)

    @staticmethod
    def get_metrics(met):
        TP, TN, FP, FN = met
        metric = OrderedDict({
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'precision': TP / max(TP + FP, 1.0e-7),
            'npv': TN / max(TN + FN, 1.0e-7),
            'recall': TP / max(TP + FN, 1.0e-7),
            'specificity': TN / max(TN + FP, 1.0e-7),
            'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
            'F1-score': 2 * TP / max(2 * TP + FP + FN, 1.0e-7),
        })
        return metric

    def optimize(self, X, y, X_test, y_test, batch_size=100, epochs=10):

        self.to(self.device)
        self.train()
        log_TM.info('\nOptimize')

        # what with weights and biases
        self.wandb.watch(self, self.loss, log='all', log_freq=100)

        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)
        self.lr_scheduler = self.lr_scheduler(self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay)

        for epoch in range(epochs):
            iterator = tqdm(
                range(0, len(X), batch_size),
                ncols=100,
                unit=f' batches - epoch:{epoch + 1}/{epochs}'
            )
            batch_list = []
            for i in iterator:
                batch_X = X[i:i + batch_size].to(self.device)
                batch_y = y[i:i + batch_size].to(self.device)
                if len(batch_y) < batch_size:
                    continue

                batch_metric, batch_loss = self.fwd_pass(batch_X, batch_y, train=True)
                batch_list.append(np.append(batch_metric['array'][:4], batch_loss.to('cpu').detach().numpy()))

                if (i > 0) and i % 1000 == 0:
                    val_metric, val_loss = self.validation(X_test, y_test)

                    batch_metric = VoodooNet.get_metrics(np.mean(batch_list, axis=0)[:4])
                    batch_loss = np.mean(batch_list, axis=0)[-1]

                    self.wandb.log({
                        'batch_metric': batch_metric, 'batch_loss': batch_loss,
                        'val_metric': val_metric,     'val_loss': val_loss
                    })

            # advance lr schedular after epoch
            self.wandb.log({'learning_rate': self.optimizer.param_groups[0]['lr']})
            self.lr_scheduler.step()

        return None

    def validation(self, X, y):
        iterator = tqdm(
            range(0, len(X), 4096),
            ncols=100,
            unit=f' batches - validation'
        )

        metrics = []
        for j in iterator:
            test_batch_X = X[j:j + 4096].to(self.device)
            test_batch_y = y[j:j + 4096].to(self.device)

            with torch.inference_mode():
                val_metric, val_loss = self.fwd_pass(test_batch_X, test_batch_y)
                metrics.append(np.append(val_metric['array'][:4], val_loss.to('cpu').detach().numpy()))

        val_metric = VoodooNet.get_metrics(np.sum(metrics, axis=0)[:4])
        val_loss = np.mean(metrics, axis=0)[-1]

        return val_metric, val_loss

    def predict(self, X_test, batch_size=4096):
        self.to(self.device)
        self.eval()
        pred = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(X_test), batch_size), ncols=100, unit=' batches'):
                batch_X = X_test[i:i + batch_size].to(self.device)
                pred.append(self(batch_X))

        return torch.cat(pred, 0)

    def save(self, path: str = None, aux: Dict = None):
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'auxiliaray': aux
        }

        os.makedirs(path[:path.rfind('/')], exist_ok=True)
        torch.save(checkpoint, path)
        self.wandb.save(path.replace('.pt', '.onnx'))

    def print_nparams(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        pytorch_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log_TM.critical(f'Total non-trainable parameters: {pytorch_total_params - pytorch_trainable_params:,d}')
        log_TM.critical(f'    Total trainable parameters: {pytorch_trainable_params:_d}')
        log_TM.critical(f'             Total  parameters: {pytorch_total_params:_d}')

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
        if string == 'crossentropy_pt':
            return nn.CrossEntropyLoss()
        elif string == 'BCEWithLogitsLoss':
            return nn.BCEWithLogitsLoss()
        elif string == 'crossentropy':
            return cross_entropy
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
    def load_pt_dataset(
            data_path: Union[str, List[str]],
            shuffle: bool = True,
            garbage: List[int] = None,
            drop_data: float = 0.0,
            dupe_CD: float = 0.0,
            groups: Dict = None,
            dy: float = 0.0,
            fn: Union[int, str, List[int]] = 0,
            site: str=''
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        def load_file(data_file):
            if os.path.isfile(data_file):
                m = torch.load(data_file)
                return m['X'], m['y']
            else:
                raise FileNotFoundError(f'"ALL-DATA-FILE" missing: {data_file}')

        # load the training data
        tread0 = time.time()

        # check input kind
        fold_list = []
        if type(fn) == int:
            fold_list = [fn]
        if fn == 'X':
            fold_list = [f'{data_path}/10foldsall-fn{ifn}-voodoo-ND2.pt' for ifn in range(10)]
        if fn == 'XX':
            fold_list = [f'{dp}/10foldsall-fn{ifn}-voodoo-ND2.pt' for ifn in range(10) for dp in data_path]
        if type(fn) == list:
            if type(fn[0]) == str:
                fold_list = fn
            else:
                fold_list = [f'{data_path}/10foldsall-fn{ifn}-voodoo-ND2.pt' for ifn in fn]

        X, y = [], []
        for ifile in tqdm(fold_list, ncols=100):
            X0, y0 = load_file(ifile)
            X.append(X0)
            y.append(y0)

        del X0, y0
        X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)

        if garbage is not None:
            for i in garbage:
                y[y == i] = 11
            X = X[y < 11]
            y = y[y < 11]

        if dupe_CD > 0:
            # lookup indices for cloud dorplet bearing classes
            idx_CD = torch.argwhere(
                torch.sum(
                    torch.stack(
                        [torch.tensor(y == i) for i in groups['0']],
                        dim=0),
                    dim=0)
            )[:, 0]
            X = torch.cat([X, torch.cat([X[idx_CD] for _ in range(int(dupe_CD))], dim=0)])
            y = torch.cat([y, torch.cat([y[idx_CD] for _ in range(int(dupe_CD))])])

        if shuffle:
            perm = torch.randperm(len(y))
            X, y = X[perm], y[perm]

        # drop some percentage from the data
        if 0 < drop_data < 1:
            idx_drop = int(X.shape[0] * drop_data)
            X, y = X[idx_drop:, ...], y[idx_drop:]

        class_counts = dict(zip(np.arange(12), np.zeros(12)))
        unique, counts = np.unique(y, return_counts=True)
        class_counts.update(dict(zip(unique, counts)))
        print(class_counts)

        tmp = torch.clone(y)
        for key, val in groups.items():  # i from 0, ..., ngroups-1
            for jclass in val:
                tmp[y == jclass] = int(key)
        y = tmp

        del tmp

        X = torch.unsqueeze(X, dim=1)
        X = torch.transpose(X, 3, 2)

        y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=len(groups)).float()
        print(f'time elapsed reading {data_path} :: {int(time.time() - tread0)} sec')

        # add delta y
        y[:, 0] = y[:, 0] - torch.full((y.size()[0],), dy, dtype=torch.float)
        y[:, 1] = y[:, 1] - torch.full((y.size()[0],), dy, dtype=torch.float)

        return X, y

    @staticmethod
    def reshape(pred, mask, newshape):
        pred_reshaped = np.zeros(newshape)
        cnt = 0
        for i in range(newshape[0]):
            for j in range(newshape[1]):
                if mask[i, j]:
                    continue
                pred_reshaped[i, j, :] = pred[cnt, :]
                cnt += 1

        return pred_reshaped


def evaluation_metrics(pred_labels, true_labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # if no status is given, metric is calculated during optimization
    for pred, truth in zip(pred_labels, true_labels):
        if truth == 1:  # TRUE
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
    from collections import OrderedDict
    out = OrderedDict({
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'precision': TP / max(TP + FP, 1.0e-7),
        'npv': TN / max(TN + FN, 1.0e-7),
        'recall': TP / max(TP + FN, 1.0e-7),
        'specificity': TN / max(TN + FP, 1.0e-7),
        'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
        'F1-score': 2 * TP / max(2 * TP + FP + FN, 1.0e-7),
    })
    out.update({
        'array': np.array([val for val in out.values()], dtype=float)
    })
    return out


def cross_entropy(input, target, size_average=False):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def smooth_prediction(probabilities, shape):
    # plot smoothed data
    from scipy.ndimage import gaussian_filter

    smoothed_probs = np.zeros(shape)
    for i in range(shape[2]):
        smoothed_probs[:, :, i] = gaussian_filter(probabilities[:, :, i], sigma=1)

    return smoothed_probs


