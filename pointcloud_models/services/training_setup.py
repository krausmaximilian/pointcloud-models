import logging

from yacs.config import CfgNode

from pointcloud_models.models import MODEL_REGISTRY
from pointcloud_models.metrics import METRIC_REGISTRY
from pointcloud_models.losses import LOSS_REGISTRY
from pointcloud_models.services.clients.mlflow_client import MlFlowClient
from pointcloud_models.utils.exceptions import TaskException, ModelNotImplementedError, LossFunctionNotImplementedError, \
    MetricNotImplementedError


# TODO add early stopping implementation
# TODO add optimizer here
# TODO add scheduler here
# TODO add dataloaders here
# TODO add checkpoint load here

class TrainingSetup:
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.model = None
        self.dataset = None
        self.criterion = None
        self.mlflow_client = None
        self.train_data_loader = None
        self.valid_data_laoder = None
        self.optimizer = None
        self.lr_scheduler = None
        self.metrics = []
        self.device = None

    def run(self):
        self.initialize_model()
        self.initialize_loss_function()
        self.initialize_metrics()
        self.initialize_train_and_validation_datasets()
        self.initialize_mlflow_client()

    def initialize_model(self):
        logging.info("Initializing model ...")
        try:
            available_networks = MODEL_REGISTRY[self.cfg.TASK.lower()]
        except KeyError:
            raise TaskException(f"Specified task {self.cfg.TASK} is invalid")
        try:
            self.model = available_networks[self.cfg.MODEL.NAME.lower()]()
        except KeyError:
            raise ModelNotImplementedError(
                f"Specified model {self.cfg.MODEL.NAME} not implemented for task {self.cfg.TASK}")

    def initialize_loss_function(self):
        logging.info("Initializing loss function ...")
        try:
            self.criterion = LOSS_REGISTRY[self.cfg.OPTIMIZER.LOSS_FUNCTION.lower()]()
        except KeyError:
            raise LossFunctionNotImplementedError(
                f"Specified loss {self.cfg.OPTIMIZER.LOSS_FUNCTION} not implemented.")

    def initialize_metrics(self):
        logging.info("Initializing metrics ...")
        for metric in self.cfg.SYSTEM.METRICS:
            try:
                self.metrics.append(METRIC_REGISTRY[metric.lower()]())
            except KeyError:
                raise MetricNotImplementedError(
                    f"Specified loss {self.cfg.OPTIMIZER.LOSS_FUNCTION} not implemented.")

    def initialize_train_and_validation_datasets(self):
        logging.info("Initializing datasets for training and validation ...")
        # TODO implement Dataset which is easily extendable
        pass

    def initialize_mlflow_client(self):
        if self.cfg.MLFLOW.ENABLED:
            logging.info("Initializing MlFlow Client ...")
            self.mlflow_client = MlFlowClient(cfg=self.cfg)
            self.mlflow_client.log_initial_params()
