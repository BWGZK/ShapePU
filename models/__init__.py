# Copyright (c) Shangqi Gao, Fudan University
from .segmentation import build


def build_model(args):
    return build(args)
