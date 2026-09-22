# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS


class FeatureType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class Feature(ABC):
    def __init__(self: Feature,
                 name: str,
                 parse_idx: int,
                 feature_type: FeatureType) -> None:
        self.type = feature_type
        self.name = name
        self.parse_idx = parse_idx

    @abstractmethod
    def domain(self: Feature) -> Union[float, np.ndarray]:
        ...

    @abstractmethod
    def map(self: Feature,
            x: str) -> Union[float, int]:
        ...


class DiscreteFeature(Feature):
    def __init__(self: DiscreteFeature,
                 name: str,
                 parse_idx: int,
                 unique_values: Sequence[str]) -> None:
        super().__init__(name, parse_idx, FeatureType.DISCRETE)
        self.unique_values: Sequence[str] = tuple(unique_values)
        self.map_to_idx: Mapping[str, int] = {x: idx for idx, x in enumerate(self.unique_values)}

    def domain(self: DiscreteFeature) -> Union[float, Sequence[str]]:
        return self.unique_values

    def map(self: DiscreteFeature,
            x: str) -> Union[float, int]:
        return self.map_to_idx[x]

    def __str__(self: DiscreteFeature) -> str:
        return f"DiscreteFeature(name={self.name}, unique_values={self.unique_values})"

    def __repr__(self: DiscreteFeature) -> str:
        return str(self)


class ContinuousFeature(Feature):
    def __init__(self: ContinuousFeature,
                 name: str,
                 parse_idx: int) -> None:
        super().__init__(name, parse_idx, FeatureType.CONTINUOUS)

    def domain(self: ContinuousFeature) -> Union[float, Sequence[str]]:
        return np.inf

    def map(self: ContinuousFeature,
            x: str) -> Union[float, int]:
        return float(x)

    def __str__(self: ContinuousFeature) -> str:
        return f"ContinuousFeature(name={self.name})"

    def __repr__(self: ContinuousFeature) -> str:
        return str(self)


class GroundTruth(DiscreteFeature):
    def __init__(self: GroundTruth,
                 unique_values: Sequence[str]) -> None:
        super().__init__("Ground Truth", -1, unique_values)

    def __str__(self: GroundTruth) -> str:
        return f"GroundTruth(unique_values={self.unique_values})"

    def __repr__(self: GroundTruth) -> str:
        return str(self)

class Header(object):

    def __init__(self: Header) -> None:
        self.features: Sequence[Feature] = list()
        self.ground_truth: GroundTruth = None

    def add_feature(self: Header,
                    feature: Feature) -> None:
        self.features.append(feature)

    def add_ground_truth(self: Header,
                         ground_truth: GroundTruth) -> None:
        self.ground_truth = ground_truth

    def parse_raw_example(self: Header,
                          raw_example: Sequence[str]) -> Tuple[Sequence[Union[float, int]], int]:
        feature_values: Sequence[Union[float, int]] = list()

        for f in self.features:
            raw_value: str = raw_example[f.parse_idx]
            feature_values.append(f.map(raw_value))

        cls = self.ground_truth.map(raw_example[self.ground_truth.parse_idx])
        return feature_values, cls

    def __len__(self: Header) -> int:
        return len(self.features)

    def __getitem__(self: Header,
                    feature_idx: int) -> Feature:
        return self.features[feature_idx]

    def __iter__(self: Header) -> Feature:
        for f in self.features:
            yield f

    def __str__(self: Header) -> str:
        return f"Header(features={self.features}, ground_truth={self.ground_truth})"

