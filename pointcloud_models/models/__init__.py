from pointcloud_models.models.pointnet import *
from pointcloud_models.models.pointnet2 import *
from pointcloud_models.models.dgcnn import *
from pointcloud_models.models.randla_net import *

from pointcloud_models.models.abstract_model import MODEL_REGISTRY, Task

__all__ = ("MODEL_REGISTRY", "Task",)
