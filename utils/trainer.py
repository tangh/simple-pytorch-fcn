import os
import time
import datetime
import math
import shutil
import logging

import tqdm
import torch
import cv2

from .misc import MetricLogger
from .metrics import label_accuracy_score
from .visualization import visualize_segmentation, get_tile_image


class Trainer(object):

    def __init__(
        self, device, model, criterion, optimizer, scheduler,
        train_loader, val_loader, save_dir, max_iter, validate_interval=None
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        if validate_interval is None:
            self.validate_interval = len(self.train_loader)
        else:
            self.validate_interval = validate_interval

        self.save_dir = save_dir
        self.max_iter = max_iter

        self.epoch = 0
        self.iteration = 0
        self.best_mean_iou = 0

    def validate(self):

        training = self.model.training
        self.model.eval()
        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []

        for batch_idx, (img, target) in tqdm.tqdm(
            enumerate(self.val_loader), total=len(self.val_loader),
            desc="val at iter=%d" % self.iteration, ncols=80, ascii=True
        ):
            img, target = img.to(self.device), target.to(self.device)
            with torch.no_grad():
                prediction = self.model(img)
            val_loss += self.criterion(prediction, target)

            # visulization
            imgs = img.data.cpu()
            lbl_true = target.data.cpu()
            lbl_pred = prediction.data.argmax(dim=1).cpu().numpy()

            for im, lt, lp in zip(imgs, lbl_true, lbl_pred):
                im, lt = self.val_loader.dataset.to_numpy(im, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = visualize_segmentation(
                        img=im, lbl_pred=lp, lbl_true=lt, n_class=n_class,
                        label_names=self.val_loader.dataset.class_names
                    )
                    visualizations.append(viz)

        metrics = label_accuracy_score(label_trues, label_preds, n_class)

        out = os.path.join(self.save_dir, 'visualization_viz')
        if not os.path.exists(out):
            os.mkdir(out)
        out_file = os.path.join(out, 'iter%012d.jpg' % self.iteration)
        cv2.imwrite(out_file, get_tile_image(visualizations)[:, :, ::-1])

        val_loss = val_loss.item() / len(self.val_loader)

        self.logger.info(
            "  ".join(["val loss: {loss}", "{meterics}"]).format(
                loss=val_loss,
                meterics=metrics
            )
        )

        # update best mIoU and save best model
        mean_iou = metrics[2]
        is_best = mean_iou > self.best_mean_iou
        if is_best:
            self.best_mean_iou = mean_iou
        checkpoint = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "arch": self.model.__class__.__name__,
            "optim_state_dict": self.optimizer.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "best_mean_iou": self.best_mean_iou,
        }
        if self.scheduler is not None:
            checkpoint.update(
                {"scheduler_state_dict": self.scheduler.state_dict()}
            )
        torch.save(checkpoint, os.path.join(self.save_dir, "checkpoint.pth.tar"))
        if is_best:
            shutil.copy(os.path.join(self.save_dir, "checkpoint.pth.tar"),
                        os.path.join(self.save_dir, "model_best.pth.tar"))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        n_class = len(self.train_loader.dataset.class_names)
        meters = MetricLogger(delimiter="  ")
        end = time.time()

        for batch_idx, (img, target) in enumerate(self.train_loader):

            # resume
            iteration = batch_idx + self.epoch * len(self.train_loader) + 1
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration

            # training
            img, target = img.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            prediction = self.model(img)
            loss = self.criterion(prediction, target)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # update meters
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                target.cpu().numpy(), prediction.argmax(dim=1).cpu().numpy(),
                n_class=n_class
            )
            current_batch_metrics = {
                "loss": loss.item(), "pixel_acc": acc,
                "mean_acc": acc_cls, "mean_iou": mean_iu, "fw_iou": fwavacc
            }
            meters.update(**current_batch_metrics)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)

            eta_seconds = meters.time.global_avg * \
                (self.max_iter - self.iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # logging
            if self.iteration % 20 == 0 or self.iteration == self.max_iter:
                self.logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}", "epoch: {epoch}", "iter: {iter}",
                            "lr: {lr:.1e}", "{meters}",
                            "current batch: {current_meters}"
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=self.epoch,
                        iter=self.iteration,
                        lr=self.optimizer.param_groups[0]["lr"],
                        meters=str(meters),
                        current_meters=str(current_batch_metrics)
                    )
                )

            # val
            if self.iteration % self.validate_interval == 0:
                self.validate()

            if self.iteration >= self.max_iter:
                if self.iteration % self.validate_interval != 0:
                    self.validate()
                break

    def train(self):
        self.logger = logging.getLogger("simple-pytorch-fcn.trainer")
        self.logger.info("Start training")

        if self.iteration == 0:
            self.validate()
            self.iteration += 1

        # resume
        if self.iteration % len(self.train_loader) == 0:
            self.epoch += 1

        max_epoch = int(math.ceil(self.max_iter / len(self.train_loader)))
        for epoch in range(self.epoch, max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
