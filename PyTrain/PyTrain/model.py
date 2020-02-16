import numpy as np

import torch
import torch.nn as nn
import torch.optim as op

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .metrics import Summary

class Model():
    def __init__(self, network, criterion, optimizer, metrics):
        self.network = network.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics

    def train(self, ds_train, ds_valid=None, epochs=1, max_steps=-1, logger=None):
        summary_train = Summary()
        summary_valid = Summary()

        metrics = self.metrics
        for epoch in range(epochs):

            self.network.train()
            metrics.begin()
            for step, (x, y) in enumerate(ds_train):
                loss, dz, dy = self._optimize(x, y)
                metrics.update(loss, dz, dy)
                if step == max_steps: break
            accuracy = metrics.commit()
            summary_train.register(epoch, loss.item(), accuracy, metrics.values)
            if not logger is None: logger.log(epoch, 'train', str(metrics))

            self.network.eval()
            metrics.begin()
            with torch.no_grad():
                for step, (dx, dy) in enumerate(ds_valid):
                    loss, dz, dy = self._validate(x, y)
                    metrics.update(loss, dz, dy)
                    if step == max_steps: break
                accuracy = metrics.commit()
                summary_valid.register(epoch, loss.item(), accuracy, metrics.values)
                if not logger is None: logger.log(epoch, 'valid', str(metrics))

    def predict(self, datasource):
        self.network.eval()
        with torch.no_grad():
            preds = []
            for step, (x, y) in enumerate(datasource):
                dz = self._forward(x)
                z = dz.detach().cpu().numpy()
                preds.append(z)
            return np.concatenate(preds, axis=0)

    def load(self, path, device=None):
        self.network.load_state_dict(torch.load(path, map_location=device))

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def _optimize(self, x, y):
        self.optimizer.zero_grad()
        dz = self._forward(x)
        dy = self._device(y, DEVICE)
        loss = self.criterion(dz, dy)
        loss.backward()
        self.optimizer.step()
        return loss, dz, dy

    def _validate(self, x, y):
        dz = self._forward(x)
        dy = self._device(y, DEVICE)
        loss = self.criterion(dz, dy)
        return loss, dz, dy

    def _forward(self, x):
        dx = self._device(x, DEVICE)
        dz = self.network(dx)
        return dz

    def _device(self, x, device):
        if isinstance(x, list):
            for n in len(x):
                x[n] = self._device(x[n], DEVICE)              
        else:
            x = x.to(DEVICE)
        return x
