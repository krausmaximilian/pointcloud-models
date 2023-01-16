import logging

from pointcloud_models.config import Config
from pointcloud_models.models import MODEL_REGISTRY
from pointcloud_models.metrics import METRIC_REGISTRY
from pointcloud_models.losses import LOSS_REGISTRY
from pointcloud_models.services.loops.training import TRAINING_LOOP_REGISTRY
from pointcloud_models.services.clients.mlflow_client import MlFlowClient
from pointcloud_models.services.model_io import ModelIO
from pointcloud_models.utils.exceptions import TaskException, ModelNotImplementedError, LossFunctionNotImplementedError, \
    MetricNotImplementedError, LoopNotImplementedError


class TrainingService:
    def __init__(self, command_line_arguments):
        self.command_line_arguments = command_line_arguments
        self.config = Config(command_line_overrides=self.command_line_arguments,
                             config_path=self.command_line_arguments.config_path)
        self.model_io = ModelIO(self.config.cfg)
        self.model = None
        self.dataset = None
        self.loss_function = None
        self.mlflow_client = None
        self.metrics = []

    def run(self):
        self.initialize_model()
        self.initialize_loss_function()
        self.initialize_metrics()
        self.initialize_train_and_validation_datasets()
        self.initialize_mlflow_client()

        # TODO load model from mlflow or local checkpoints
        self.initialize_training()

    def initialize_model(self):
        logging.info("Initializing model ...")
        try:
            available_networks = MODEL_REGISTRY[self.config.cfg.TASK.lower()]
        except KeyError:
            raise TaskException(f"Specified task {self.config.cfg.TASK} is invalid")
        try:
            self.model = available_networks[self.config.cfg.MODEL.NAME.lower()]()
        except KeyError:
            raise ModelNotImplementedError(
                f"Specified model {self.config.cfg.MODEL.NAME} not implemented for task {self.config.cfg.TASK}")

    def initialize_loss_function(self):
        logging.info("Initializing loss function ...")
        try:
            self.loss_function = LOSS_REGISTRY[self.config.cfg.OPTIMIZER.LOSS_FUNCTION.lower()]()
        except KeyError:
            raise LossFunctionNotImplementedError(
                f"Specified loss {self.config.cfg.OPTIMIZER.LOSS_FUNCTION} not implemented.")

    def initialize_metrics(self):
        logging.info("Initializing metrics ...")
        for metric in self.config.cfg.SYSTEM.METRICS:
            try:
                self.metrics.append(METRIC_REGISTRY[metric.lower()]())
            except KeyError:
                raise MetricNotImplementedError(
                    f"Specified loss {self.config.cfg.OPTIMIZER.LOSS_FUNCTION} not implemented.")

    def initialize_train_and_validation_datasets(self):
        logging.info("Initializing datasets for training and validation ...")
        pass

    def initialize_mlflow_client(self):
        if self.config.cfg.MLFLOW.ENABLED:
            logging.info("Initializing MlFlow Client ...")
            self.mlflow_client = MlFlowClient(config=self.config.cfg)
            self.mlflow_client.log_initial_params()

    def initialize_training(self):
        logging.info("Initializing training ...")
        try:
            TRAINING_LOOP_REGISTRY[self.config.cfg.TASK]()
        except KeyError:
            raise LoopNotImplementedError(f"No training loop implemented for task {self.config.cfg.TASK}")
