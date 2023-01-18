from abc import ABC, abstractmethod

from pointcloud_models.config import Config
from pointcloud_models.services.model_io import ModelIO
from pointcloud_models.services.training_setup import TrainingSetup

TRAINING_REGISTRY = {}


def register_class(cls):
    TRAINING_REGISTRY[cls.__name__] = cls


def factory(name):
    return TRAINING_REGISTRY[name]()


class AbstractTrainingService(ABC):
    def __init__(self, command_line_arguments):
        super().__init__()
        self.command_line_arguments = command_line_arguments
        self.config = Config(command_line_overrides=self.command_line_arguments,
                             config_path=self.command_line_arguments.config_path)

        self.model_io = ModelIO(self.config.cfg)
        self.training_setup = TrainingSetup(self.config.cfg)
        self.training_setup.run()

    def __init_subclass__(cls, task=None, **kwargs):
        super().__init_subclass__(**kwargs)
        register_class(cls)

    @abstractmethod
    def training_loop(self):
        pass
