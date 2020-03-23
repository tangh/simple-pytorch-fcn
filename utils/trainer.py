import os
import math
import shutil

import fcn
import torch
import skimage
import numpy as np

from .metrics import label_accuracy_score


class Trainer(object):

    def __init__(
        self, device, model, criterion, optimizer, train_loader, val_loader,
        save_dir, max_iter, validate_interval=None
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
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

        for batch_idx, (img, target) in enumerate(self.val_loader):
            img, target = img.to(self.device), target.to(self.device)
            with torch.no_grad():
                prediction = self.model(img)

            # loss = cross_entropy2d(score, target,
            #                        size_average=self.size_average)
            # loss_data = loss.data.item()
            # if np.isnan(loss_data):
            #     raise ValueError('loss is nan while validating')
            # val_loss += loss_data / len(data)
            loss = self.criterion(prediction, target)
            val_loss += loss

            imgs = img.data.cpu()
            lbl_pred = prediction.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for im, lt, lp in zip(imgs, lbl_true, lbl_pred):
                im, lt = self.val_loader.dataset.to_numpy(im, lt)

                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=im, n_class=n_class)
                    visualizations.append(viz)
            if batch_idx % 200 == 0:
                print("val on batch id:", batch_idx)
        metrics = label_accuracy_score(label_trues, label_preds, n_class)

        out = os.path.join(self.save_dir, 'visualization_viz')
        if not os.path.exists(out):
            os.mkdir(out)
        out_file = os.path.join(out, 'iter%012d.jpg' % self.iteration)
        skimage.io.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)
        print("metrics and loss", metrics, val_loss.item())

        # with open(osp.join(self.out, 'log.csv'), 'a') as f:
        #     elapsed_time = (
        #         datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
        #         self.timestamp_start).total_seconds()
        #     log = [self.epoch, self.iteration] + [''] * 5 + \
        #           [val_loss] + list(metrics) + [elapsed_time]
        #     log = map(str, log)
        #     f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iou
        if is_best:
            self.best_mean_iou = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iou': self.best_mean_iou,
        }, os.path.join(self.save_dir, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(os.path.join(self.save_dir, 'checkpoint.pth.tar'),
                        os.path.join(self.save_dir, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        n_class = len(self.train_loader.dataset.class_names)
        running_loss = 0

        for batch_idx, (img, target) in enumerate(self.train_loader):

            # resume
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration

            # val
            if self.iteration % self.validate_interval == 0:
                self.validate()

            # training
            img, target = img.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            # score = self.model(data)
            prediction = self.model(img)
            # loss = cross_entropy2d(score, target,
            #                        size_average=self.size_average)
            # loss /= len(data)
            # loss_data = loss.data.item()
            # if np.isnan(loss_data):
            #     raise ValueError('loss is nan while training')
            loss = self.criterion(prediction, target)
            loss.backward()
            self.optimizer.step()

            metrics = []
            # lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            # lbl_true = target.data.cpu().numpy()
            # acc, acc_cls, mean_iu, fwavacc = \
            #     torchfcn.utils.label_accuracy_score(
            #         lbl_true, lbl_pred, n_class=n_class)
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                target.cpu().numpy(), prediction.max(dim=1)[1].cpu().numpy(),
                n_class=n_class
            )
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            running_loss += loss.item()
            if batch_idx % 20 == 0:
                print("Training on [epoch{}/iter{}]: loss/average loss = {}/{}, \
                    metrics[acc,acc_cls,miou,fwavacc] = {}".format(
                    self.epoch, self.iteration,
                    loss.item(), running_loss/(batch_idx+1), metrics.tolist()
                ))

            # with open(osp.join(self.out, 'log.csv'), 'a') as f:
            #     elapsed_time = (
            #         datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
            #         self.timestamp_start).total_seconds()
            #     log = [self.epoch, self.iteration] + [loss_data] + \
            #         metrics.tolist() + [''] * 5 + [elapsed_time]
            #     log = map(str, log)
            #     f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(self.max_iter / len(self.train_loader)))
        for epoch in range(self.epoch, max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
