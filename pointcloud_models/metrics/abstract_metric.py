from abc import ABC, abstractmethod
from enum import Enum
import torch.nn as nn


METRIC_REGISTRY = {}


def register_class(cls, task):
    METRIC_REGISTRY[cls.__name__] = cls


def factory(name, task):
    return METRIC_REGISTRY[name]()
