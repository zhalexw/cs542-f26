# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence, Set
from typing import Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS
from .data.header import Feature, FeatureType, ContinuousFeature, DiscreteFeature, Header
from .quality.quality_function import QualityFunction
from .model import Model


class Node(ABC):
    def __init__(self: Node,
                 header: Header,
                 quality_function: QualityFunction,
                 X: np.ndarray,
                 y_gt: np.ndarray) -> None:
        self.header: Header = header
        self.quality_function = quality_function

        # some stats
        self.num_samples: int = X.shape[0]
        self.unique_classes, self.unique_class_counts = np.unique(y_gt, return_counts=True)

        # majority class
        self.majority_class = self.get_majority_class(X, y_gt)

        # children of this Node
        self.children = list()

    def get_majority_class(self: Node,
                           X: np.ndarray,
                           y_gt: np.ndarray) -> int:
        unique_classes, counts = np.unique(y_gt, return_counts=True)
        counts_argmax = np.argmax(counts)
        return unique_classes[counts_argmax]

    @abstractmethod
    def predict(self: Node,
                x: np.ndarray) -> Union[int, Node]:
        ...

    @abstractmethod
    def get_child_datasets(self: Node) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        ...

    @abstractmethod
    def is_leaf(self: Node) -> bool:
        ...

    @abstractmethod
    def _to_str(self: Node,
                depth: int) -> str:
        ...

    def __str__(self: Node) -> str:
        return self._to_str(0)

    def __repr__(self: Node) -> str:
        return str(self)


class LeafNode(Node):
    def __init__(self: Node,
                 header: Header,
                 quality_function: QualityFunction,
                 X: np.ndarray,
                 y_gt: np.ndarray) -> None:
        super().__init__(header, quality_function, X, y_gt)

    def predict(self: LeafNode,
                x: np.ndarray) -> Union[int, Node]:
        return self.majority_class

    def get_child_datasets(self: LeafNode) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        return None

    def is_leaf(self: LeafNode) -> bool:
        return True

    def _to_str(self: LeafNode,
                depth: int) -> str:
        return ("\t" * depth) + f"LeafNode(majority_class={self.majority_class})"


class InteriorNode(Node):
    def __init__(self: InteriorNode,
                 header: Header,
                 quality_function: QualityFunction,
                 X: np.ndarray,
                 y_gt: np.ndarray,
                 available_feature_idxs: Set[int]) -> None:
        super().__init__(header, quality_function, X, y_gt)

        # these will be set by self._pick_best_feature
        self.feature_quality: float = None
        self.feature_idx: int = None
        self.feature_split_values: Sequence[float] = None
        self.feature_type: FeatureType = None

        # the feature idxs that children can use
        self.child_feature_idxs = set(available_feature_idxs)

        # call this to choose which feature this InteriorNode will focus on
        self._pick_best_feature(X, y_gt, available_feature_idxs)

        # remember to delete the chosen feature from child datasets if we choose a discrete feature!
        if self.feature_type == FeatureType.DISCRETE:
            self.child_feature_idxs.remove(self.feature_idx)


    def _pick_best_feature(self: InteriorNode,
                           X: np.ndarray,
                           y_gt: np.ndarray,
                           available_feature_idxs: Set[int]) -> int:
        best_feature_quality = -np.inf
        best_feature_idx: int = -1
        best_feature_split_values: np.ndarray = None

        # argmax the features
        for feature_idx in available_feature_idxs:
            feature_quality, feature_split_values = self._eval_feature(X, y_gt, feature_idx)

            # argmax and settle ties with the smallest feature_idx
            if (feature_quality > best_feature_quality) or (feature_quality == best_feature_quality and
                                                            feature_idx < best_feature_idx):
                best_feature_quality = feature_quality
                best_feature_idx = feature_idx
                best_feature_split_values = feature_split_values

        # set the fields
        self.feature_quality = best_feature_quality
        self.feature_idx = best_feature_idx
        self.feature_split_values = best_feature_split_values
        self.feature_type = self.header[self.feature_idx].type

    def _eval_feature(self: InteriorNode,
                      X: np.ndarray,
                      y_gt: np.ndarray,
                      feature_idx: int) -> tuple[float, Sequence[float]]:
        feature_quality: float = None
        feature_split_values: Sequence[float] = list()

        X_col: np.ndarray = X[:, feature_idx]
        feature_type = self.header[feature_idx].type
        features = np.unique(X_col)

        
        if feature_type == FeatureType.DISCRETE:
            feature_quality = self.quality_function.quality(y_gt, [y_gt[X_col == val] for val in features])
            feature_split_values = features

        elif feature_type == FeatureType.CONTINUOUS:
            thresholds = self._get_continuous_feature_thresholds(X_col, y_gt)

            best_quality = -np.inf
            for threshold in thresholds:
                left_y_gt = y_gt[X_col <= threshold]
                right_y_gt = y_gt[X_col > threshold]
                quality = self.quality_function.quality(y_gt, [left_y_gt, right_y_gt])
                if quality > best_quality:
                    best_quality = quality
                    feature_quality = quality
                    feature_split_values = [threshold] #one split value t: <=t and >t

        return feature_quality, feature_split_values

    def _get_continuous_feature_thresholds(self: InteriorNode,
                                           X_col: np.ndarray,
                                           y_gt: np.ndarray) -> Sequence[float]:
        thresholds: Sequence[float] = list()

        #iterate through cont x array
        #if current gt ≠ next gt, is threshold and add average to thresholds
        for i in range(X_col.shape[0] - 1):
            if y_gt[i] != y_gt[i + 1]:
                thresholds.append((X_col[i] + X_col[i + 1]) / 2)

        return thresholds

    def predict(self: InteriorNode,
                x: np.ndarray) -> Union[int, Node]:
        
        child: Node = None
        feature = x[self.feature_idx]

        for i, split_value in enumerate(self.feature_split_values):
            if self.feature_type == FeatureType.DISCRETE:
                if feature == split_value:
                    child = self.children[i]
                    break
            elif self.feature_type == FeatureType.CONTINUOUS:
                if feature <= split_value:
                    child = self.children[i]
                    break
                else:
                    child = self.children[i + 1]
                    break

        return child

    def get_child_datasets(self: InteriorNode,
                           X: np.ndarray,
                           y_gt: np.ndarray) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        child_datasets: Sequence[tuple[np.ndarray, np.ndarray]] = list()

        # get the column of data that this interior node focuses on
        X_col: np.ndarray = X[:, self.feature_idx]
        feature_type = self.header[self.feature_idx].type

        if feature_type == FeatureType.DISCRETE:
            for split_value in self.feature_split_values:
                child_X = X[X_col == split_value]
                child_y_gt = y_gt[X_col == split_value]
                child_datasets.append((child_X, child_y_gt))
        elif feature_type == FeatureType.CONTINUOUS:
            threshold = self.feature_split_values[0]
            left_X = X[X_col <= threshold]
            left_y_gt = y_gt[X_col <= threshold]
            right_X = X[X_col > threshold]
            right_y_gt = y_gt[X_col > threshold]
            child_datasets.append((left_X, left_y_gt))
            child_datasets.append((right_X, right_y_gt))

        return child_datasets

    def is_leaf(self: InteriorNode) -> bool:
        return False

    # helpful for printing a decision tree
    def _to_str(self: InteriorNode,
                depth: int) -> str:
        feature_name: str = self.header[self.feature_idx].name
        split_values: str = self.feature_split_values

        preamble = "InteriorNode("
        children = "children=["

        return ("\t" * depth) + f"{preamble}feature={feature_name}, split_values={split_values})" +\
            "\n" + "\n".join([c._to_str(depth+1) for c in self.children])


class DecisionTreeClassifier(Model):
    def __init__(self: DecisionTreeClassifier,
                 header: Header,
                 quality_function: QualityFunction,
                 available_feature_idxs: Set[int] = None) -> None:
        self.header: Header = header
        self.quality_function = quality_function
        self.available_feature_idxs: Set[int] = set(available_feature_idxs) \
                                                if available_feature_idxs is not None \
                                                else set(range(len(self.header)))

        self.num_nodes: int = 0
        self.root: Node = None

    def _build(self: DecisionTreeClassifier,
               X: np.ndarray,
               y_gt: np.ndarray,
               available_feature_idxs: Set[int],
               depth: int,
               pre_prune_function: Callable[[np.ndarray, np.ndarray, Set[int], int], bool] = None) -> Node:
        node: Node = None

        # TODO: build the tree! This method needs to turn a dataset into a node.
        #       If that node is an InteriorNode we need to get the child datasets and
        #       turn them into nodes too!
        #
        #       The argument 'pre_prune_function' will not be 'None' if the caller requests pre-pruning
        #       to occur. Pre-pruning is something like setting a "max depth" or "minimum samples", you don't
        #       have to care because this is a function pointer. If this argument is not 'None' and you call
        #       it, it will return 'true' when you should clip this path and generate a LeafNode (even if
        #       an InteriorNode would be chosen otherwise).
        #
        #       you should expect this to be called like this:
        #           pre_prune_function(X, y_gt, available_feature_idxs, depth)

        return node

    def fit(self: DecisionTreeClassifier,
            X: np.ndarray,
            y_gt: np.ndarray,
            pre_prune_function: Callable[[np.ndarray, np.ndarray, Set[int], int], bool] = None,
            mcc_prune: bool = False,
            alpha: float = 0.5) -> None:
        # alpha is the hyperparameter coefficient for the pessimistic error estimate

        # build the tree
        self.root = self._build(X, y_gt, self.available_feature_idxs, 1, pre_prune_function=pre_prune_function)

        # TODO: implement minimum-cost-complexity pruning algorithm!

    def _predict_sample(self: DecisionTreeClassifier,
                        x: np.ndarray) -> int:
        node: Node = self.root
        while not node.is_leaf():
            node = node.predict(x)

        # node should be a leaf node
        return node.predict(x)

    def predict(self: DecisionTreeClassifier,
                X: np.ndarray) -> np.ndarray:

        # pre-allocate space for each prediction
        num_samples: int = X.shape[0]
        y_hat = np.zeros(num_samples)

        # one sample at a time
        for sample_idx in range(num_samples):
            y_hat[sample_idx] = self._predict_sample(X[sample_idx, :])
        return y_hat

    def __str__(self: DecisionTreeClassifier) -> str:
        return str(self.root)

    def __repr__(self: DecisionTreeClassifier) -> str:
        return str(self)



"""
    A bagging model. This model will train a bunch of decision trees and then have each of its internal
    decision trees vote during inference. The class that gets the most votes wins (settling ties arbitrarily).
"""
class RandomForestClassifier(Model):
    def __init__(self: RandomForestClassifier,
                 header: Header,
                 quality_function: QualityFunction,
                 max_num_features: int = None,
                 num_trees: int = 100) -> None:
        self.header = header
        self.quality_function = quality_function
        self.max_num_features = min(len(self.header), max_num_features) \
                                if max_num_features is not None else len(self.header)
        self.num_trees = num_trees
        self.trees: Sequence[Model] = list()

    def _bootstrap_sample(self: RandomForestClassifier,
                          X: np.ndarray,
                          y_gt: np.ndarray,
                          num_samples) -> tuple[np.ndarray, np.ndarray]:
        # draw num_samples uniformly and independently at random
        sample_idxs: np.ndarray = np.random.choice(X.shape[0], size=num_samples, replace=True)

        # note that this **applies** the indices to the data 
        return X[sample_idxs, :], y_gt[sample_idxs]

    def _sample_features(self: RandomForestClassifier,
                         max_features: int) -> Set[int]:
        # choose 'max_features' from the available features
        num_features: int = np.random.choice(max_features)

        # note that this returns a **set** of feature indices
        return set(np.random.choice(len(self.header), size=num_features, replace=False))

    def fit(self: RandomForestClassifier,
            X: np.ndarray,
            y_gt: np.ndarray,
            pre_prune_function: Callable[[np.ndarray, np.ndarray, Set[int], int], bool] = None,
            mcc_prune: bool = False,
            alpha: float = 0.5) -> None:
        """
            train 'num_trees' number of DecisionTreeClassifiers
            each tree gets its own dataset:
                - the samples in that dataset are bootstrap sampled. This means that you sample
                  uniformly and independently at random a bunch of samples from X (the same sample
                  can appear multiple times within the generated dataset)
                - the features for that specific dataset are restricted to be only a subset of the
                  original features. The features a specific tree is allowed to use are chosen
                  uniformly at random (not independently though: no duplicate features!). There
                  is a parameter called 'max_num_features' which gives an upper bound on how
                  many features a single tree can see.
        """

        # TODO: build the forest! 
        ...

    def predict(self: RandomForestClassifier,
                X: np.ndarray) -> np.ndarray:
        # TODO: ask each tree to predict 'X' and then implement majority voting!
        ...

    # helpful for printing the forest
    def __str__(self: RandomForestClassifier) -> str:
        return f"RandomForestClassifier(header={self.header}, self.quality_function={type(self.quality_function).__name__}, max_num_features={self.max_num_features}, num_trees={self.num_trees})"

    def __repr__(self: RandomForestClassifier) -> str:
        return str(self)

