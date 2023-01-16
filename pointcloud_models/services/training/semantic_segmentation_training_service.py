import logging

import torch

from pointcloud_models.services.training import AbstractTrainingService


class SemanticSegmentationTrainingService(AbstractTrainingService):
    def training_loop(self):
        def bn_momentum_adjust(m, momentum):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.momentum = momentum

        LEARNING_RATE_CLIP = 1e-5
        MOMENTUM_ORIGINAL = 0.1
        MOMENTUM_DECAY = 0.5
        MOMENTUM_DECAY_STEP = 10

        # TODO getting start epoch from checkpoint
        start_epoch = 0
        max_iou = 0
        best_model = self.training_setup.model

        for epoch in range(start_epoch, self.config.cfg.OPTIMIZER.EPOCHS):
            num_batches = len(self.training_setup.train_data_loader)

            total_correct = 0
            total_seen = 0
            loss_sum = 0

            # logger.info("\nEpoch: {}".format(epoch))
            momentum = MOMENTUM_ORIGINAL * (
                    MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP)
            )
            if momentum < 0.01:
                momentum = 0.01
            # logger.info("BN momentum updated to: %f" % momentum)
            model = model.apply(lambda x: bn_momentum_adjust(x, momentum))

            for i, (points, labels) in enumerate(self.training_setup.train_data_loader):
                model = model.train()
                points = points.data.numpy()
                # TODO Randomly rotated point cloud around z axis

                points = torch.Tensor(points)

                if self.training_setup.device != torch.device("cpu"):
                    points, labels = points.float().cuda(device=self.training_setup.device), labels.long().cuda(
                        device=self.training_setup.device)
                else:
                    points, labels = points.float(), labels.long()

                points = points.transpose(2, 1)
                self.training_setup.optimizer.zero_grad()

                predictions = model(points)

                predictions = predictions.contiguous().view(-1, self.config.cfg.DATASET.NUM_CLASSES)
                # batch_label = labels.view(-1, 1)[:, 0].cpu().data.numpy()
                labels = labels.view(-1, 1)[:, 0]
                loss = self.training_setup.criterion(seg_pred, labels)
                loss.backward()
                self.training_setup.optimizer.step()

                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += c.OPTIMIZER.batch_size * c.DATASET.num_points
                loss_sum += loss
            logger.info("Training mean loss: %f" % (loss_sum / num_batches).item())
            logger.info("Training accuracy: %f" % (total_correct / float(total_seen)))

            if c.MLFLOW.active:
                mlflow.log_metric(
                    "learning_rate", optimizer.param_groups[0]["lr"], step=epoch
                )
                mlflow.log_metric("train_loss", (loss_sum / num_batches).item(), step=epoch)
                mlflow.log_metric(
                    "train_accuracy", total_correct / float(total_seen), step=epoch
                )

            # validation epcoh
            with torch.no_grad():
                model = model.eval()
                num_batches = len(valid_data_loader)
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                labelweights = np.zeros(c.DATASET.num_classes)
                total_seen_class = [0 for _ in range(c.DATASET.num_classes)]
                total_correct_class = [0 for _ in range(c.DATASET.num_classes)]
                total_iou_deno_class = [0 for _ in range(c.DATASET.num_classes)]
                for i, (points, labels) in enumerate(valid_data_loader, 1):
                    points = points.data.numpy()
                    points = torch.Tensor(points)
                    if device != torch.device("cpu"):
                        points, labels = points.float().cuda(device=device), labels.long().cuda(device=device)
                    else:
                        points, labels = points.float(), labels.long()
                    points = points.transpose(2, 1)
                    seg_pred, trans_feat = model(points)
                    pred_val = seg_pred.contiguous().cpu().data.numpy()
                    seg_pred = seg_pred.contiguous().view(-1, c.DATASET.num_classes)
                    batch_label = labels.cpu().data.numpy()
                    labels = labels.view(-1, 1)[:, 0]
                    loss = criterion(seg_pred, labels)
                    loss_sum += loss
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum((pred_val == batch_label))
                    total_correct += correct
                    total_seen += c.OPTIMIZER.batch_size * c.DATASET.num_points
                    tmp, _ = np.histogram(batch_label, range(c.DATASET.num_classes + 1))
                    labelweights += tmp
                    for l in range(c.DATASET.num_classes):
                        total_seen_class[l] += np.sum((batch_label == l))
                        total_correct_class[l] += np.sum(
                            (pred_val == l) & (batch_label == l)
                        )
                        total_iou_deno_class[l] += np.sum(
                            ((pred_val == l) | (batch_label == l))
                        )
                labelweights = labelweights.astype(np.float32) / np.sum(
                    labelweights.astype(np.float32)
                )
                mIoU = np.mean(
                    np.array(total_correct_class)
                    / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
                )
                logger.info("eval mean loss: %f" % (loss_sum / float(num_batches)).item())
                logger.info("eval point avg class IoU: %f" % (mIoU))
                logger.info("eval point accuracy: %f" % (total_correct / float(total_seen)))
                logger.info(
                    "eval point avg class acc: %f"
                    % (
                        np.mean(
                            np.array(total_correct_class)
                            / (np.array(total_seen_class, dtype=np.float) + 1e-6)
                        )
                    )
                )
                iou_per_class_str = "------- IoU --------\n"
                for l in range(c.DATASET.num_classes):
                    iou_per_class_str += "class %s weight: %.3f, IoU: %.3f \n" % (
                        str(l) + " ",
                        labelweights[l],
                        total_correct_class[l] / float(total_iou_deno_class[l]),
                    )

                logger.info(iou_per_class_str)

                if c.MLFLOW.active:
                    mlflow.log_metric("valid_iou", mIoU, step=epoch)
                    mlflow.log_metric(
                        "valid_loss", (loss_sum / float(num_batches)).item(), step=epoch
                    )
                    mlflow.log_metric(
                        "valid_accuracy", total_correct / float(total_seen), step=epoch
                    )
                    for r in range(c.DATASET.num_classes):
                        mlflow.log_metric(
                            "valid_iou_c" + str(r),
                            total_correct_class[r] / float(total_iou_deno_class[r]),
                            step=epoch,
                        )

            scheduler.step((loss_sum / float(num_batches)).item())

            if mIoU > max_iou:
                max_iou = mIoU
                best_model = model
                save_checkpoint(
                    best_model, c, epoch, optimizer, (loss_sum / float(num_batches)).item()
                )

            early_stopping((loss_sum / float(num_batches)).item(), model)
            if early_stopping.early_stop:
                logger.info("Early Stopping")
                break

        # save best model for inference
        save_model(best_model, c, max_iou)
        if c.MLFLOW.active:
            mlflow.end_run()
