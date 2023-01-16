from yacs.config import CfgNode


class ModelIO:
    def __init__(self, config: CfgNode):
        self.config = config

    def save_model(self):
        pass

    def save_checkpoint(self):
        pass

    def load_model(self):
        pass

    def load_checkpoint(self):
        pass
