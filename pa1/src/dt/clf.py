# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence, Set
from copy import copy
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

        self.X = X
        self.y_gt = y_gt

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
            if feature_quality is None:
                feature_quality = -np.inf

        return feature_quality, feature_split_values

    def _get_continuous_feature_thresholds(self: InteriorNode,
                                           X_col: np.ndarray,
                                           y_gt: np.ndarray) -> Sequence[float]:
        thresholds: Sequence[float] = list()

        #iterate through cont x array
        #if current gt ≠ next gt, is threshold and add average to thresholds
        sorted_idxs = np.argsort(X_col)
        sorted_X_col = X_col[sorted_idxs]
        sorted_y_gt = y_gt[sorted_idxs]

        for i in range(sorted_X_col.shape[0] - 1):
            if sorted_y_gt[i] != sorted_y_gt[i + 1] and sorted_X_col[i] != sorted_X_col[i + 1]:
                thresholds.append((sorted_X_col[i] + sorted_X_col[i + 1]) / 2)

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

        return child if child is not None else LeafNode(
    self.header, self.quality_function, self.X, self.y_gt
)

    def get_child_datasets(self: InteriorNode,
                           X: np.ndarray = None,
                           y_gt: np.ndarray = None) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        child_datasets: Sequence[tuple[np.ndarray, np.ndarray]] = list()

        # get the column of data that this interior node focuses on
        X_col: np.ndarray = X[:, self.feature_idx]
        feature_type = self.header[self.feature_idx].type

        if feature_type == FeatureType.DISCRETE:
            for split_value in self.feature_split_values:
                child_X = X[X_col == split_value]
                child_y_gt = y_gt[X_col == split_value]
                child_datasets.append((child_X, child_y_gt))
                #print(f"Child dataset for split value {split_value}: {child_X.shape[0]} samples")
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
        

        should_pre_prune = pre_prune_function is not None and pre_prune_function(X, y_gt, available_feature_idxs, depth)

        if (len(available_feature_idxs) == 0) or (len(np.unique(y_gt)) == 1) or should_pre_prune:
            self.num_nodes += 1
            return LeafNode(self.header, self.quality_function, X, y_gt)

        else: 
            node = InteriorNode(self.header, self.quality_function, X, y_gt, available_feature_idxs)
            if node.feature_idx < 0 or len(node.feature_split_values) == 0:
                self.num_nodes += 1
                return LeafNode(self.header, self.quality_function, X, y_gt)

            self.num_nodes += 1
            child_datasets = node.get_child_datasets(X, y_gt)
            for child_X, child_y_gt in child_datasets:
                child_node = self._build(child_X, child_y_gt, set(node.child_feature_idxs), depth + 1, pre_prune_function)
                node.children.append(child_node)

            return node

    def _leaf_error(self: DecisionTreeClassifier,
                    node: Node) -> int:
        return node.num_samples - int(np.max(node.unique_class_counts))

    def _tree_stats(self: DecisionTreeClassifier,
                    node: Node) -> tuple[int, int]:
        if node.is_leaf():
            return self._leaf_error(node), 1

        num_errors = 0
        num_leaves = 0
        for child in node.children:
            child_errors, child_leaves = self._tree_stats(child)
            num_errors += child_errors
            num_leaves += child_leaves
        return num_errors, num_leaves

    def _pessimistic_error(self: DecisionTreeClassifier,
                           node: Node,
                           alpha: float) -> float:
        num_errors, num_leaves = self._tree_stats(node)
        return num_errors + alpha * num_leaves

    def _clone_tree(self: DecisionTreeClassifier,
                    node: Node) -> Node:
        """Clone the tree structure while sharing its immutable training arrays."""
        cloned_node = copy(node)
        cloned_node.children = [self._clone_tree(child) for child in node.children]
        return cloned_node

    def _minimum_cost_complexity_candidate(
            self: DecisionTreeClassifier,
            root: Node,
            alpha: float) -> tuple[Node, int, InteriorNode]:
        """Find the interior node selected by the Task 6 pruning ratio."""
        best_parent = None
        best_child_idx = -1
        best_node = None
        best_ratio = np.inf

        def visit(node: Node,
                  parent: Node = None,
                  child_idx: int = -1) -> tuple[int, int]:
            nonlocal best_parent, best_child_idx, best_node, best_ratio

            if node.is_leaf():
                return self._leaf_error(node), 1

            subtree_errors = 0
            subtree_leaves = 0
            for idx, child in enumerate(node.children):
                child_errors, child_leaves = visit(child, node, idx)
                subtree_errors += child_errors
                subtree_leaves += child_leaves

            pruned_errors = self._leaf_error(node)
            error_difference = (pruned_errors - subtree_errors) + \
                               alpha * (1 - subtree_leaves)
            removed_leaves = subtree_leaves - 1
            ratio = error_difference / removed_leaves if removed_leaves > 0 else 0.0

            if ratio < best_ratio:
                best_parent = parent
                best_child_idx = child_idx
                best_node = node
                best_ratio = ratio

            return subtree_errors, subtree_leaves

        visit(root)
        return best_parent, best_child_idx, best_node

    def _minimum_cost_complexity_prune(self: DecisionTreeClassifier,
                                       root: Node,
                                       alpha: float) -> Node:
        """Build the pruning sequence and return its minimum-error tree."""
        current_root = root
        best_root = self._clone_tree(current_root)
        best_error = self._pessimistic_error(current_root, alpha)

        while not current_root.is_leaf():
            parent, child_idx, node_to_prune = \
                self._minimum_cost_complexity_candidate(current_root, alpha)
            pruned_node = LeafNode(self.header,
                                   self.quality_function,
                                   node_to_prune.X,
                                   node_to_prune.y_gt)

            if parent is None:
                current_root = pruned_node
            else:
                parent.children[child_idx] = pruned_node

            current_error = self._pessimistic_error(current_root, alpha)
            if current_error < best_error:
                best_error = current_error
                best_root = self._clone_tree(current_root)

        return best_root

    def _count_nodes(self: DecisionTreeClassifier,
                     node: Node) -> int:
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def fit(self: DecisionTreeClassifier,
            X: np.ndarray,
            y_gt: np.ndarray,
            pre_prune_function: Callable[[np.ndarray, np.ndarray, Set[int], int], bool] = None,
            mcc_prune: bool = False,
            alpha: float = 0.5) -> None:
        # alpha is the hyperparameter coefficient for the pessimistic error estimate

        # build the tree
        self.num_nodes = 0
        self.root = self._build(X, y_gt, self.available_feature_idxs, 1, pre_prune_function=pre_prune_function)

        if mcc_prune:
            self.root = self._minimum_cost_complexity_prune(self.root, alpha)
            self.num_nodes = self._count_nodes(self.root)


    def _predict_sample(self: DecisionTreeClassifier,
                        x: np.ndarray) -> int:
        node: Node = self.root
        while not node.is_leaf():
            next_node = node.predict(x)
            if not isinstance(next_node, Node):
                return next_node
            node = next_node

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

