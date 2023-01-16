from abc import ABC, abstractmethod
from enum import Enum
import torch.nn as nn


class Task(str, Enum):
    SEMANTIC_SEGMENTATION = "SEMANTIC_SEGMENTATION"
    PART_SEGMENTATION = "PART_SEGMENTATION"
    OBJECT_CLASSIFICATION = "OBJECT_CLASSIFICATION"

    @classmethod
    def get_all_members(cls):
        return [member for member in cls.__members__.values()]


MODEL_REGISTRY = {t: {} for t in Task.get_all_members()}


def register_class(cls, task):
    MODEL_REGISTRY[task][cls.__name__] = cls


def factory(name, task):
    return MODEL_REGISTRY[task][name]()


class AbstractModel(ABC, nn.Module):
    def __int__(self, num_classes: int, input_dimensions: int):
        super().__init__()
        self.input_dimensions = input_dimensions
        self.num_classes = num_classes

    def __init_subclass__(cls, task=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if task not in Task.get_all_members():
            raise ValueError(f"{task} is not defined for Point Cloud Models")
        register_class(cls, task)
