import logging

import pkbar
import torch

from pointcloud_models.metrics import get_segmentation_statistics
from pointcloud_models.services.training import AbstractTrainingService
from pointcloud_models.metrics.functional_segmentation_metrics import iou_score, accuracy


class SemanticSegmentationTrainingService(AbstractTrainingService):
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = 10

    def training_loop(self):
        def bn_momentum_adjust(m, momentum):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.momentum = momentum

        # TODO getting start epoch from checkpoint
        start_epoch = 0
        max_iou = 0
        best_model = self.training_setup.model

        for epoch in range(start_epoch, self.config.cfg.OPTIMIZER.EPOCHS):
            num_batches = len(self.training_setup.train_data_loader)
            training_bar = pkbar.Kbar(
                target=num_batches,
                epoch=epoch,
                num_epochs=self.config.cfg.OPTIMIZER.EPOCHS,
                stateful_metrics=[
                    "train_mean_loss",
                    "train_mean_acc",
                    "train_mean_iou",
                ],
            )
            training_loss_sum = 0
            momentum = self.MOMENTUM_ORIGINAL * (
                    self.MOMENTUM_DECAY ** (epoch // self.MOMENTUM_DECAY_STEP)
            )
            if momentum < 0.01:
                momentum = 0.01
            logging.info("BN momentum updated to: %f" % momentum)
            self.training_setup.model = self.training_setup.model.apply(lambda x: bn_momentum_adjust(x, momentum))

            mean_train_accuracy_average_class = 0
            mean_train_iou_average_class = 0

            for i, (points, labels) in enumerate(self.training_setup.train_data_loader):
                self.training_setup.model = self.training_setup.model.train()
                tp, fp, fn, tn, training_loss_sum, loss = self._train(points, labels, training_loss_sum)
                average_iou_per_batch = iou_score(tp, fp, fn, tn, reduction="macro")
                average_accuracy_per_batch = accuracy(tp, fp, fn, tn, reduction="macro")

                mean_train_accuracy_average_class += average_accuracy_per_batch
                mean_train_iou_average_class += average_iou_per_batch

                training_bar.update(
                    i,
                    values=[
                        ("train_mean_loss", training_loss_sum.item()),
                        ("train_mean_acc", mean_train_accuracy_average_class / (i + 1)),
                        ("train_mean_iou", mean_train_iou_average_class / (i + 1))
                    ],
                )

            logging.info(f"Training mean loss: {(training_loss_sum / num_batches).item()}")
            logging.info(f"Training point avg class accuracy: {mean_train_accuracy_average_class / num_batches}")
            logging.info(f"Training point avg class IoU: {mean_train_iou_average_class / num_batches}")

            # TODO log current learning rate, current loss, current accuracy, current iou to mlflow if active

            # validation epoch
            with torch.no_grad():
                num_batches = len(self.training_setup.valid_data_loader)
                validation_bar = pkbar.Kbar(
                    target=num_batches,
                    epoch=epoch,
                    num_epochs=self.config.cfg.OPTIMIZER.EPOCHS,
                    stateful_metrics=[
                        "validation_mean_loss",
                        "validation_mean_acc",
                        "validation_mean_iou",
                    ],
                )
                self.training_setup.model = self.training_setup.model.eval()
                validation_loss_sum = 0
                mean_validation_accuracy_average_class, mean_validation_iou_average_class = 0, 0
                per_class_validation_iou = torch.zeros(self.config.cfg.DATASET.NUM_CLASSES)
                for i, (points, labels) in enumerate(self.training_setup.valid_data_loader, 1):
                    tp, fp, fn, tn, validation_loss_sum, loss = self._validation(points, labels, validation_loss_sum)
                    average_iou_per_batch = iou_score(tp, fp, fn, tn, reduction="macro")
                    per_class_iou_per_batch = iou_score(tp, fp, fn, tn, reduction=None).sum(0)
                    average_accuracy_per_batch = accuracy(tp, fp, fn, tn, reduction="macro")

                    mean_validation_accuracy_average_class += average_accuracy_per_batch
                    mean_validation_iou_average_class += average_iou_per_batch
                    per_class_validation_iou += per_class_iou_per_batch

                    validation_bar.update(
                        i,
                        values=[
                            ("validation_mean_loss", validation_loss_sum.item() / (i+1)),
                            ("validation_mean_acc", mean_validation_accuracy_average_class / (i + 1)),
                            ("validation_mean_iou", mean_validation_iou_average_class / (i + 1))
                        ],
                    )

                result_mean_iou_average_class = mean_validation_iou_average_class / num_batches
                result_mean_loss = (validation_loss_sum / num_batches).item()
                result_mean_accuracy_average_class = mean_validation_accuracy_average_class / num_batches
                result_per_class_validation_iou = per_class_validation_iou / num_batches
                # TODO add normal accuracy, not average of class, total and log
                logging.info(f"validation mean loss: {result_mean_loss}")
                logging.info(f"validation point avg class IoU: {result_mean_iou_average_class}")
                logging.info(f"validation point avg class accuracy: {result_mean_accuracy_average_class}")
                iou_per_class_str = "------- IoU --------\n"
                for index in range(per_class_validation_iou):
                    iou_per_class_str += f"class {index}, IoU: {per_class_validation_iou[index].item()} \n"
                logging.info(iou_per_class_str)

                # TODO if mlflow is active, log mean iou, average class acc, acc, and iou per class

            # TODO consider using something else than loss here?
            self.training_setup.scheduler.step(result_mean_loss)

            if result_mean_iou_average_class > max_iou:
                max_iou = result_mean_iou_average_class
                best_model = self.training_setup.model
                # TODO save checkpoint and model

            # TODO early stopping

        # save best model for inference
        # TODO save the best model

    def _train(self, points, labels, loss_sum):
        # TODO Randomly rotated point cloud around z axis
        points = torch.Tensor(points)
        points, labels = points.float().to(device=self.training_setup.device), labels.long().to(
            device=self.training_setup.device)
        points = points.transpose(2, 1)
        self.training_setup.optimizer.zero_grad()
        predictions = self.training_setup.model(points)
        predictions_for_loss = predictions.contiguous().view(-1, self.training_setup.cfg.DATASET.NUM_CLASSES)
        labels_for_loss = labels.view(-1, 1)[:, 0]

        loss = self.training_setup.criterion(predictions_for_loss, labels_for_loss)
        loss.backward()
        loss_sum += loss

        self.training_setup.optimizer.step()

        predicted_labels = predictions.data.max(1)[1]
        tp, fp, fn, tn = get_segmentation_statistics(predicted_labels, labels)
        return tp, fp, fn, tn, loss_sum, loss

    def _validation(self, points, labels, loss_sum):
        points = torch.Tensor(points)
        points, labels = points.float().to(device=self.training_setup.device), labels.long().to(
            device=self.training_setup.device)
        points = points.transpose(2, 1)
        predictions = self.training_setup.model(points)
        predictions_for_loss = predictions.contiguous().view(-1, self.training_setup.cfg.DATASET.NUM_CLASSES)
        labels_for_loss = labels.view(-1, 1)[:, 0]
        loss = self.training_setup.criterion(predictions_for_loss, labels_for_loss)
        loss_sum += loss
        predicted_labels = predictions.data.max(1)[1]
        tp, fp, fn, tn = get_segmentation_statistics(predicted_labels, labels)
        return tp, fp, fn, tn, loss_sum, loss
