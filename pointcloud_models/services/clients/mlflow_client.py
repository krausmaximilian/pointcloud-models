import os
import mlflow
from yacs.config import CfgNode


class MlFlowClient:
    def __init__(self, config: CfgNode):
        self.config = config
        os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW.USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW.PASSWORD
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = config.MLFLOW.IGNORE_SSL_VERIFY
        mlflow.set_tracking_uri(config.MLFLOW.TRACKING_URL)
        self.experiment = None

    def __del__(self):
        mlflow.end_run()

    def start_run(self):
        mlflow.set_experiment(self.config.MLFLOW.EXPERIMENT_NAME)
        self.experiment = mlflow.get_experiment_by_name(self.config.MLFLOW.EXPERIMENT_NAME)
        mlflow.start_run(experiment_id=self.experiment.experiment_id)

    def log_initial_params(self):
        # mlflow.log_param()
        # mlflow.log_params()
        # TODO
        pass

    def log_model(self, model):
        """# creating model signature
        # TODO
        input = mlflow.types.TensorSpec(
            type=np.dtype("f"),
            shape=[-1, data_dim, c.DATASET.img_size, c.DATASET.img_size],
        )
        output = mlflow.types.TensorSpec(type=np.dtype("f"), shape=[-1, num_classes])
        input_schema = mlflow.types.Schema([input])
        output_schema = mlflow.types.Schema([output])
        signature = mlflow.models.ModelSignature(
            inputs=input_schema, outputs=output_schema
        )
        # TODO requirements file
        mlflow.pytorch.log_model(model, "models", signature=signature)
        mlflow.log_metric("best_valid_iou", best_iou)
        mlflow.log_param(
            "local save path",
            local_save_path + "/" + time + ".pth",
        )

        # also log checkpoint, so we can us it for training in future
        # also log requirements
        mlflow.log_artifact("requirements.txt")"""
        pass

    def log_checkpoint(self):
        # mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")
        # TODO
        pass

    def load_model(self, model_uri: str, dst_path=None):
        """Loads model from MlFlow, S3 or local path. Run_id currently not supported yet."""
        return mlflow.pytorch.load_model(model_uri)

    def load_checkpoint(self):
        # TODO
        pass
