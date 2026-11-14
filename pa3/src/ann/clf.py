# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence, Set
from typing import Tuple, Union
from tqdm import tqdm
import copy as cp
import numpy as np
import sys


# PYTHON PROJECT IMPORTS
from .activations.activation import Activation
from .activations.sigmoid import Sigmoid
from .activations.softmax import Softmax
from .data.header import Feature, FeatureType, ContinuousFeature, DiscreteFeature, Header
from .losses.bce import BinaryCrossEntropy
from .losses.cce import CategoricalCrossEntropy
from .losses.lossfunction import LossFunction
from .model import Model
from .optimizers.optimizer import Optimizer
from .parameter import Parameter


class ANNC(Model):
    def __init__(self: ANNC,
                 header: Header,
                 num_hidden_units: int,
                 hidden_activation: Activation) -> None:
        self.header: Header = header
        self.num_hidden_units: int = num_hidden_units
        self.hidden_activation: Activation = hidden_activation

        # predict Pr[class1] if 2 output and a probability distribution over the classes if not binary classification
        self.num_outputs = 1 if len(self.header.ground_truth.domain()) == 2 else len(self.header.ground_truth.domain())

        # if only 2 classes use sigmoid (and BCE) and otherwise Softmax (and CCE)
        self.output_activation: Activation = Sigmoid() if self.num_outputs == 1 else Softmax()
        self.loss_function: LossFunction = BinaryCrossEntropy() if self.num_outputs == 1 else CategoricalCrossEntropy()

        # 
        self.num_inputs, self.input_to_feature_maps = self._allocate_inputs(self.header)

        # input to hidden layer parameters
        self.W1 = Parameter(np.random.randn(self.num_inputs, self.num_hidden_units))
        self.b1 = Parameter(np.random.randn(1, self.num_hidden_units))

        # hidden to output layer parameters
        self.W2 = Parameter(np.random.randn(self.num_hidden_units, self.num_outputs))
        self.b2 = Parameter(np.random.randn(1, self.num_outputs))

    def parameters(self: ANNC) -> Sequence[Parameter]:
        return [self.W1, self.b1, self.W2, self.b2]

    """
        needs to calculate how many inputs to this ANN samples will have:
            - if the feature is continuous then the network should allocate one input unit for that feature
            - if the feature is discrete convert that feature into a "one-hot" encoding:
                - if a feature has 'k' values, then allocate 'k' input units for that feature.
                - when presented with a value, map that value to a "one-hot" binary vector representing *which*
                  value (of the k possible) was present

        For instance, lets say your header has the following features in the following order:
                feature 0                       feature 1                               feature 2
            ContinuousFeature()     DiscreteFeature(domain=['a', 'b', 'c'])     DiscreteFeature(domain=['a', 'e'])

        Then you would want your neural network to allocate:
                1 input unit                    3 input units                           2 input units

        The network is making this choice so it needs to wrap around it (i.e. convert between samples provided
        by the user and the input features expected by the network. For instance, lets say a sample was provided
        by the user that had the following raw values:
            (6.2, 'b', 'e')

        Really the values given to the model for this sample would be:
            (6.1, 1, 1)
        because the Header will use the DiscreteFeature type to map raw values into some integer values.

        Either way, the neural network is expecting the sample to look like this:

             continous value    one-hot for feature 1       one-hot for feature 2
            (6.2,                   0, 1, 0,                        0, 1)

        And the ANN needs to keep these decision consistent. So, the purpose of this method is to do two things:
            - calculate the number of inputs the network will expect samples to be described in
            - calculate whatever values are needed to convert from user-provided samples into network-expected samples

        I have chosen these necessary values to be a list of maps (i.e. a list of dictionaries).
        For each user-provided feature, this method will calculate a dictionary which will map user-provded values
        of that feature into the feature expected by the network. The purpose of building a list of them is that
        we *don't* want to allocate a dictionary for continuous features and we *do* want to allocate a dictionary
        for discrete features.

        For instance, back to our original example. If the header has the following features in the following order:
                feature 0                       feature 1                               feature 2
            ContinuousFeature()     DiscreteFeature(domain=['a', 'b', 'c'])     DiscreteFeature(domain=['a', 'e'])

        then this method should return the following pair:
            - 6 (the number of inputs should be 1 + |D_f1| + |D_f2| = 1 + 3 + 2 = 6)
            - [None,                                             (no map for continuous features)
               {f1.map('a'): 0, f1.map('b'): 1, f1.map('c'): 2}, (maps "indices f1.map produces" to "network inputs")
               {f2.map('a'): 0, f2.map('e'): 1}                  (maps "indices f2.map produces" to "network inputs")
              ]
    """
    def _allocate_inputs(self: ANNC, header: Header) -> tuple[int, Sequence[Mapping[float, int]]]:
        num_inputs: int = 0
        input_to_feature_maps: Sequence[Mapping[float, int]] = list()

        for feature in self.header.features:
            input_to_feature_map: Mapping[float, int] = None

            if feature.type == FeatureType.DISCRETE:
                num_inputs += len(feature.domain())
                input_to_feature_map = {float(feature.map(raw_val)): idx
                                        for idx, raw_val in enumerate(feature.domain())}
            else:
                num_inputs += 1
            input_to_feature_maps.append(input_to_feature_map)
        return num_inputs, input_to_feature_maps

    """
        A method to actually transform each sample in 'X_user_samples' to the feature representation
        that the neural network expects (see '_allocate_inputs'). If 'X_user_samples' has 'n' rows and 'm'
        columns this method should return a matrix with 'n' rows and 'self.num_inputs' columns where the value
        of each column (for each row) is set according to the description in '_allocate_inputs').
    """
    def _map_user_samples(self: ANNC,
                          X_user_samples: np.ndarray) -> np.ndarray:
        # TODO
        ...

    """
        A method to calculate the gradients for 'W' and 'b' within a single layer. This method expects
        the inputs 'X' to the layer and the gradient of the loss function with respect to the outputs
        of the layer 'dLoss_dActivation_output.' This method needs to increment the gradients cached
        in the 'Parameter' class (i.e. 'W.val += ...' and 'b.val += ...')
    """
    def _update_layer(self: ANNC,
                      X: np.ndarray,
                      dLoss_dActivation_output: np.ndarray,
                      W: Parameter,
                      b: Parameter,
                      activation: Activation) -> np.ndarray:
        # TODO
        ...

    def _backward(self: ANNC,
                  X_network_samples: np.ndarray,
                  Y_gt: np.ndarray,
                  gamma: float = 0.0) -> None:
        A0: np.ndarray = X_network_samples

        # input to hidden layer
        Z1: np.ndarray = np.dot(A0, self.W1.val) + self.b1.val
        A1: np.ndarray = self.hidden_activation.forward(Z1)

        # hidden to output layer
        Z2: np.ndarray = np.dot(A1, self.W2.val) + self.b2.val
        A2: np.ndarray = self.output_activation.forward(Z2)

        # predictions to loss_function
        dLoss_dA2: np.ndarray = self.loss_function.backward(A2, Y_gt)

        # update the hidden -> output layer
        dLoss_dA1: np.ndarray = self._update_layer(A1, dLoss_dA2,
                                                   self.W2, self.b2, self.output_activation)

        # update the input -> hidden layer
        dLoss_dA0: np.ndarray = self._update_layer(A0, dLoss_dA1,
                                                   self.W1, self.b1, self.hidden_activation)

        # weight decay terms...weight decay is adding the following penalty term onto your objective:
        #
        #       gamma * sum{p \in parameters} p^2
        #
        # which has derivative:
        #
        #       2 * gamma * p
        for P in self.parameters():
            P.grad += 2*gamma*P.val

    def fit(self: ANNC,
            X_user_samples: np.ndarray,
            y_gt: np.ndarray,
            optimizer: Optimizer,
            gamma: float = 0.0,
            epsilon: float = 1e-8,
            max_iters: int = int(1e6)) -> None:

        # need to map 'X_user_samples' into samples the network expects
        X_network_samples: np.ndarray = self._map_user_samples(X_user_samples)
        Y_gt: np.ndarray = y_gt.reshape(-1,1) # make this a column vector

        # perform gradient descent until convergence

        # TODO
        ...

    def _predict(self: ANNC,
                 X_network_samples: np.ndarray) -> np.ndarray:
        # input to hidden dimension
        Z1: np.ndarray = np.dot(X_network_samples, self.W1.val) + self.b1.val
        A1: np.ndarray = self.hidden_activation.forward(Z1)

        # hidden to output dimension
        Z2: np.ndarray = np.dot(A1, self.W2.val) + self.b2.val
        return self.output_activation.forward(Z2)

    def predict_proba(self: ANNC,
                      X_user_samples: np.ndarray) -> np.ndarray:
        # needs to convert user-features into features the network expects
        return self._predict(self._map_user_samples(X_user_samples))

    def predict(self: ANNC,
                X_user_samples: np.ndarray) -> np.ndarray:
        y_probs: np.ndarray = self.predict_proba(X_user_samples)

        # return whichever class is the most likely
        return (y_probs >= 0.5).reshape(-1).astype(int) if self.num_outputs == 1 else y_probs.argmax(axis=1)

    def __str__(self: ANNC) -> str:
        return f"ANN(header={self.header}, dims=[{self.num_inputs}->{self.num_hidden_units}->{self.num_outputs}]"\
            +f" activations=[{self.hidden_activation.__class__.__name__}, {self.output_activation.__class__.__name__}])"

    def __repr__(self: ANNC) -> str:
        return str(self)

