import numpy as np

import torch
import torch.nn as nn
import torch.optim as op

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .metrics import Summary

class Model():
    def __init__(self, network, criterion, optimizer, scheduler, metrics):
        self.network = network.to(DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

    def train(self, ds_train, ds_valid=None, epochs=1, max_steps=-1, logger=None):
        summary_train = Summary()
        summary_valid = Summary()
        for epoch in range(epochs):
            self._train_epoch(epoch, ds_train, max_steps, logger, summary_train)
            if ds_valid:
                self._valid_epoch(epoch, ds_valid, max_steps, logger, summary_valid)
        return summary_train, summary_valid

    def _train_epoch(self, epoch, ds_train, max_steps, logger, summary_train):
        metrics = self.metrics
        metrics.begin()
        self.network.train()
        for step, (x, y) in enumerate(ds_train):
            loss, dz, dy = self._optimize(x, y)
            metrics.update(loss, dz, dy)
            if step == max_steps: break
        if self.scheduler: self.scheduler.step()
        accuracy = metrics.commit(epoch)
        summary_train.register(epoch, loss.item(), accuracy, metrics.values)
        if logger: logger.log(epoch, str(metrics), 'train')

    def _valid_epoch(self, epoch, ds_valid, max_steps, logger, summary_valid):
        metrics = self.metrics
        metrics.begin()
        self.network.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(ds_valid):
                loss, dz, dy = self._validate(x, y)
                metrics.update(loss, dz, dy)
                if step == max_steps: break
            accuracy = metrics.commit(epoch)
            summary_valid.register(epoch, loss.item(), accuracy, metrics.values)
            if logger: logger.log(epoch, str(metrics), 'valid')

    def predict(self, dataloader):
        self.network.eval()
        with torch.no_grad():
            preds = []
            for step, (x, y) in enumerate(dataloader):
                dz = self._forward(x)
                z = dz.detach().cpu().numpy()
                preds.append(z)
            return np.concatenate(preds, axis=0)

    def load(self, path, device=DEVICE):
        self.network.load_state_dict(torch.load(path, map_location=device))

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def _optimize(self, x, y):
        self.optimizer.zero_grad()
        dz = self._forward(x)
        dy = self._device(y)
        loss = self.criterion(dz, dy)
        loss.backward()
        self.optimizer.step()
        return loss, dz, dy

    def _validate(self, x, y):
        dz = self._forward(x)
        dy = self._device(y)
        loss = self.criterion(dz, dy)
        return loss, dz, dy

    def _forward(self, x):
        dx = self._device(x)
        dz = self.network(dx)
        return dz

    def _device(self, x):
        if isinstance(x, list):
            for n in len(x):
                x[n] = self._device(x[n])
        else:
            x = x.to(DEVICE)
        return x
