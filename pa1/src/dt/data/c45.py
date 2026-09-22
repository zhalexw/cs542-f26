# SYSTEM IMPORTS
from collections.abc import Sequence, Set
from typing import Tuple
import numpy as np
import os
import sys


# _cd_ = os.path.abspath(os.path.dirname(__file__))
# if _cd_ not in sys.path:
#     sys.path.append(_cd_)
# del _cd_


# PYTHON PROJECT IMPORTS
from .header import ContinuousFeature, DiscreteFeature, FeatureType, Feature, GroundTruth, Header


def parse_c45(names_filepath: str,
              data_filepath: str,
              ignore_feature_names: Set[str] = None) -> Tuple[Header, np.ndarray, np.ndarray]:
    if not os.path.exists(names_filepath):
        raise ValueError(f"ERROR: {names_filepath} does not exist!")
    if not os.path.exists(data_filepath):
        raise ValueError(f"ERROR: {data_filepath} does not exist!")

    header: Header = _parse_names(names_filepath, ignore_feature_names=ignore_feature_names)
    X, y_gt = _parse_data(header, data_filepath, ignore_feature_names=ignore_feature_names)
    return header, X, y_gt


def _parse_names(names_filepath: str,
                 ignore_feature_names: Set[str] = None) -> Header:
    lines: Sequence[str] = list()
    with open(names_filepath, "r") as f:
        for line in f:
            if len(line.strip().rstrip()) > 0:
                lines.append(line.strip().rstrip())

    header: Header = Header()

    # class has to be the first line. Comma separated and ends with a "."
    class_line: str = lines[0]
    if class_line.endswith("."):
        class_line = class_line[:-1]
    classes = sorted(set([x.strip().rstrip() for x in class_line.split(",")]))
    header.add_ground_truth(GroundTruth(classes))

    # features are the rest of the lines
    feature_idx: int = 0
    for line in lines[1:]:
        if ":" in line and line.endswith("."):
            feature_name = line.split(":")[0]
            if feature_name not in ignore_feature_names:
                header.add_feature(_parse_feature(line, feature_idx))
            feature_idx += 1

    return header


def _parse_feature(feature_line: str,
                   feature_idx: int) -> Feature:
    feature_line = feature_line[:-1] # remove trailing '.'
    line_parts = feature_line.split(":")
    feature_name = line_parts[0].rstrip().rstrip()
    feature_values = line_parts[1].strip().rstrip()

    # print(feature_line, line_parts)

    feature: Feature = None
    if feature_values.startswith("continuous"):
        feature = ContinuousFeature(feature_name, feature_idx)
    elif "," in feature_values:
        unique_values = sorted(set([x.strip().rstrip() for x in feature_values.split(",")]))
        feature = DiscreteFeature(feature_name, feature_idx, unique_values)
    else:
        raise ValueError(f"ERROR: unknown feature line [{feature_line}]")
    return feature


def _parse_data(header: Header,
                data_filepath: str,
                ignore_feature_names = None) -> Tuple[np.ndarray, np.ndarray]:
    examples: Sequence[Sequence[Union[float, int]]] = list()
    ground_truth: Sequence[int] = list()

    num_lines = 0
    with open(data_filepath, "r") as f:
        for line in f:
            num_lines += 1
            line = line.strip().rstrip()

            # each line should have len(header) + 1 features
            example = [x.strip().rstrip() for x in line.split(",")]
            # print(len(example), len(header))

            example, cls = header.parse_raw_example(example)

            ground_truth.append(cls)
            examples.append(example)

    return np.array(examples), np.array(ground_truth)


def load_spam() -> Tuple[Header, np.ndarray, np.ndarray]:
    cd = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.join(cd, "spam")
    names_path = os.path.join(root_dir, "spam.names")
    data_path = os.path.join(root_dir, "spam.data")

    ignore_features = ["index"]
    return parse_c45(names_path, data_path, ignore_feature_names=ignore_features)


def load_voting() -> Tuple[Header, np.ndarray, np.ndarray]:
    cd = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.join(cd, "voting")
    names_path = os.path.join(root_dir, "voting.names")
    data_path = os.path.join(root_dir, "voting.data")

    ignore_features = ["index"]
    return parse_c45(names_path, data_path, ignore_feature_names=ignore_features)


def load_volcanoes() -> Tuple[Header, np.ndarray, np.ndarray]:
    cd = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.join(cd, "volcanoes")
    names_path = os.path.join(root_dir, "volcanoes.names")
    data_path = os.path.join(root_dir, "volcanoes.data")

    ignore_features = ["index", "image_id"]
    return parse_c45(names_path, data_path, ignore_feature_names=ignore_features)


def load_play_outside() -> tuple[Header, np.ndarray, np.ndarray]:
    cd = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.join(cd, "play_outside")
    names_path = os.path.join(root_dir, "play_outside.names")
    data_path = os.path.join(root_dir, "play_outside.data")

    ignore_features = ["index"]
    return parse_c45(names_path, data_path, ignore_feature_names=ignore_features)


if __name__ == "__main__":
    header, X, y_gt = load_spam()
    print("spam:")
    print(len(header), X.shape, y_gt.shape, np.unique(y_gt, return_counts=True)[1], header)
    print()
    print()

    header, X, y_gt = load_voting()
    print("voting:")
    print(len(header), X.shape, y_gt.shape, np.unique(y_gt, return_counts=True)[1], header)
    print()
    print()

    header, X, y_gt = load_volcanoes()
    print("volcanoes:")
    print(len(header), X.shape, y_gt.shape, np.unique(y_gt, return_counts=True)[1], header)
    print()
    print()

    header, X, y_gt = load_play_outside()
    print("play outside:")
    print(len(header), X.shape, y_gt.shape, np.unique(y_gt, return_counts=True)[1], header)

