from pointcloud_models.models import Task
from pointcloud_models.services.loops.training.train_semantic_segmentation import train_semantic_segmentation

TRAINING_LOOP_REGISTRY = {Task.SEMANTIC_SEGMENTATION: train_semantic_segmentation}


__all__ = ("TRAINING_LOOP_REGISTRY", )
