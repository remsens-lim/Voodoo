"""
This module contains functions for generating deep learning models with Tensorflow and Keras.

"""

import os
from collections import OrderedDict

import numpy as np

np.random.seed(0)

import torch
import torch.nn as nn
torch.manual_seed(0)

from tqdm.auto import tqdm
from typing import Tuple, Dict, List


__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2021, The Voodoo Project"
__credits__ = ["Willi Schimmel"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


class Conv2DUnit(nn.Module):
    def __init__(self, in_shape, nfltrs, kernel, stride, padding, downsample=None):
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
        #output = self.pool(output)

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
            dev: str = 'cpu',
            loss: str = None,
            resnet: bool = False,
            fn: int = 0,
            p: float=0.5,
            batch_size=256,
            metrics: List[str] = None,
            kernel_init: str = None,
            regularizer: str = None,
            batch_norm: str = True,
            hidden_activations: str = 'elu',
            output_activation: str = None,
            dropout: float = 0.0):
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
        self.hidden_activations = hidden_activations
        self.regularizer = regularizer
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
            self.pad_sizes = np.array([[kern[0]-1, kern[1]-2] for kern in kernel_sizes])

        self.num_filters = np.array(num_filters)
        self.dense_layers = np.array(dense_layers)
        self.dropout = dropout
        self.device = dev
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
        self.define_cnn()

        # flatten last convolution
        self.flatten = nn.Flatten()

        # calculate flat dimension
        self.n_flat = self.flatten_conv()

        # initialize dense layers
        self.define_dense(dropout=dropout)

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
        self.dense.update({f'dense_{i + 1}': nn.Linear(in_shape, self.output_shape)})
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
        metrics = []
        loss = self.loss(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return metrics, loss

    def predict(self, X_test, batch_size=2048):
        self.to(self.device)
        self.eval()
        pred = []
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test), batch_size), ncols=100, unit=' batches'):
                batch_X = X_test[i:i + batch_size].to(self.device)
                pred.append(self(batch_X))

        activation = self.output_activation(dim=1)
        return activation(torch.cat(pred, 0))

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
        print(f'Total non-trainable parameters: {pytorch_total_params - pytorch_trainable_params:,d}')
        print(f'    Total trainable parameters: {pytorch_trainable_params:_d}')
        print(f'             Total  parameters: {pytorch_total_params:_d}')

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
            print(f'Loaded model from disk:\n{model_path}')
            return torch.load(model_path)
        else:
            raise RuntimeError(f'No model found for path: {model_path}')
        
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

def identity(input):
    return input

def smooth_prediction(probabilities, shape):
        # plot smoothed data
        from scipy.ndimage import gaussian_filter

        smoothed_probs = np.zeros(shape)
        for i in range(shape[2]):
            smoothed_probs[:, :, i] = gaussian_filter(probabilities[:, :, i], sigma=1)

        return smoothed_probs


