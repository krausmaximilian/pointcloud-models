from abc import ABC, abstractmethod
from enum import Enum
import torch.nn as nn


LOSS_REGISTRY = {}


def register_class(cls, task):
    LOSS_REGISTRY[cls.__name__] = cls


def factory(name, task):
    return LOSS_REGISTRY[name]()
