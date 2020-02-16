import numpy as np

class Summary():
    def __init__(self):
        self.epoch = -1
        self.loss = None
        self.accuracy = None
        self.values = None
        self.history = []

    def register(self, epoch, loss, accuracy, values={}):
        values['epoch'] = epoch
        values['loss'] = loss
        values['accuracy'] = accuracy
        self.history.append(values)
        if self.epoch < 0 or accuracy > self.accuracy:
            self.epoch = epoch
            self.loss = loss
            self.accuracy = accuracy
            self.values = values
            return True
        return False

    def hash(self):
        def r4(v): return "{:6.4f}".format(v)
        return F'{self.epoch:02}-{r4(self.accy).strip()}'
    
class BaseMetrics():
    def __init__(self):
        self.loss = None
        self.preds = None
        self.targets = None
        self.values = None

    def begin(self):
        self.loss = []
        self.preds = []
        self.targets = []
        self.values = None

    def update(self, loss, dz, dy):
        self.loss.append(self._get_loss(loss))
        self.preds.append(self._get_preds(dz))
        self.targets.append(self._get_targets(dy))

    def _get_loss(self, loss):
        return loss.item()

    def _get_preds(self, dz):
        return dz.detach().cpu().numpy()

    def _get_targets(self, dy):
        return dy.detach().cpu().numpy()

    def commit(self):
        loss = np.mean(self.loss)
        accuracy = 0.0
        self.values = {
                'loss': 0.0,
                'accuracy': 0.0
            }
        return accuracy

    def __str__(self):
        def r4(v): return "{:6.4f}".format(v)
        return F'{r4(self.values["loss"])}\t{r4(self.values["accuracy"])}'

class ClassificationMetrics(BaseMetrics):
    def _get_preds(self, dz):
        preds = dz.max(axis=1)[1]
        return preds.detach().cpu().numpy()

    def _get_targets(self, dy):
        return dy.detach().cpu().numpy()

    def commit(self):
        preds = np.concatenate(self.preds)
        targets = np.concatenate(self.targets)
        loss = np.mean(self.loss)
        accuracy = sum(preds == targets) / len(preds)
        self.values = {
                'loss': loss,
                'accuracy': accuracy
            }
        return accuracy
