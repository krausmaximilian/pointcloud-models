import os

from yacs.config import CfgNode as Cn, CfgNode
import typing as t


class Config:
    def __init__(self, config_path: t.Optional[str], command_line_overrides: t.Optional[list]):
        self._C = Cn()
        self._initialize_settings()
        self.config_path = config_path
        self.command_line_overrides = command_line_overrides
        self.cfg = self.construct_settings()

    def _initialize_settings(self):
        self._C.DATASET = Cn()
        self._C.TASK = os.getenv("TASK", "semantic_segmentation")

        self._C.DATASET.NUM_CLASSES = os.getenv("NUM_CLASSES", None)
        self._C.DATASET.DATA_DIMENSION = os.getenv("DATA_DIMENSION", None)
        self._C.DATASET.DATASET_TYPE = os.getenv("DATASET_TYPE", None)

        self._C.SYSTEM = Cn()
        self._C.SYSTEM.NUM_WORKERS = os.getenv("NUM_WORKERS", 4)
        self._C.SYSTEM.USE_GPU = os.getenv("USE_GPU", False)
        self._C.SYSTEM.GPU_ID = os.getenv("GPU_ID", 0)
        self._C.SYSTEM.METRICS = os.getenv("METRICS", ["iou", ])

        self._C.MODEL = Cn()
        self._C.MODEL.NAME = os.getenv("MODEL_NAME", "pointnet")
        self._C.MODEL.FINAL_ACTIVATION = os.getenv("FINAL_ACTIVATION", "softmax2d")
        self._C.MODEL.IOU_THRESHOLD = os.getenv("IOU_THRESHOLD", 0.5)

        self._C.OPTIMIZER = Cn()
        self._C.OPTIMIZER.NAME = os.getenv("OPTIMIZER_NAME", "adam")
        self._C.OPTIMIZER.LEARNING_RATE_SCHEDULER = os.getenv("OPTIMIZER_NAME", "ReduceLROnPlateau")
        self._C.OPTIMIZER.LEARNING_RATE = os.getenv("LEARNING_RATE", 0.0001)
        self._C.OPTIMIZER.EPOCHS = os.getenv("EPOCHS", 20)
        self._C.OPTIMIZER.BATCH_SIZE = os.getenv("BATCH_SIZE", 10)
        self._C.OPTIMIZER.EARLY_STOPPING_ENABLED = os.getenv("EARLY_STOPPING_ENABLED", False)
        self._C.OPTIMIZER.EARLY_STOPPING_PATIENCE = os.getenv("EARLY_STOPPING_PATIENCE", 10)
        self._C.OPTIMIZER.LOSS_FUNCTION = os.getenv("LOSS_FUNCTION", "NLLLoss")

        self._C.MLFLOW = Cn()
        self._C.MLFLOW.ENABLED = os.getenv("MLFLOW_ENABLED", False)
        self._C.MLFLOW.TRACKING_URL = os.getenv("MLFLOW_TRACKING_URL", None)
        self._C.MLFLOW.USERNAME = os.getenv("MLFLOW_USERNAME", None)
        self._C.MLFLOW.PASSWORD = os.getenv("MLFLOW_PASSWORD", None)
        self._C.MLFLOW.EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
        self._C.MLFLOW.IGNORE_SSL_VERIFY = os.getenv("MLFLOW_IGNORE_SSL_VERIFY", False)

    def get_cfg_defaults(self) -> CfgNode:
        """Get a yacs CfgNode object with default values for project"""
        # Return a clone so that defaults will not be altered
        return self._C.clone()

    def construct_settings(self) -> CfgNode:
        cfg = self.get_cfg_defaults()
        if self.config_path and os.path.isfile(self.config_path):
            cfg.merge_from_file(self.config_path)

        # Now override from a list (opts could come from the command line)
        # opts = ["SYSTEM.NUM_GPUS", 8, "TRAIN.SCALES", "(1, 2, 3, 4)"]
        if self.command_line_overrides:
            cfg.merge_from_list(self.command_line_overrides)
        return cfg
