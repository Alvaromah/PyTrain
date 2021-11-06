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

    def __str__(self):
        def r4(v): return "{:6.4f}".format(v)
        return F'best epoch: {self.epoch}\tloss:\t{r4(self.loss)}\taccy:\t{r4(self.accuracy)}'
    
class BaseMetrics():
    def __init__(self):
        self.loss = None
        self.preds = None
        self.targs = None
        self.values = None

    def begin(self):
        self.loss = []
        self.preds = []
        self.targs = []
        self.values = None

    def update(self, loss, dz, dy):
        self.loss.append(self._get_loss(loss))
        self.preds.append(self._get_preds(dz))
        self.targs.append(self._get_targets(dy))

    def _get_loss(self, loss):
        return loss.item()

    def _get_preds(self, dz):
        return dz.detach().cpu().numpy()

    def _get_targets(self, dy):
        return dy.detach().cpu().numpy()

    def commit(self, epoch):
        loss = np.mean(self.loss)
        accuracy = 0.0
        self.values = {
                'loss': loss,
                'accuracy': accuracy
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

    def commit(self, epoch):
        loss = np.mean(self.loss)
        preds = np.concatenate(self.preds)
        targs = np.concatenate(self.targs)
        accuracy = sum(preds == targs) / len(preds)
        self.values = {
                'loss': loss,
                'accuracy': accuracy
            }
        return accuracy

class CrossEntropyMetrics(BaseMetrics):
    def commit(self, epoch):
        loss = np.mean(self.loss)
        preds = np.concatenate(self.preds, axis=0)
        targs = np.concatenate(self.targs, axis=0)

        preds = preds.argmax(axis=1)
        targs = targs.argmax(axis=1)
        preds = preds.reshape(preds.shape[0], -1)
        targs = targs.reshape(targs.shape[0], -1)
        accuracy = (sum(preds == targs) / len(preds)).mean()

        self.values = {
                'loss': loss,
                'accuracy': accuracy
            }
        return accuracy

class RMSEMetrics(BaseMetrics):
    def commit(self, epoch):
        loss = np.mean(self.loss)
        preds = np.concatenate(self.preds, axis=0)
        targs = np.concatenate(self.targs, axis=0)

        preds = preds.reshape(preds.shape[0], -1)
        targs = targs.reshape(targs.shape[0], -1)
        accuracy = np.sqrt(np.mean((targs - preds) ** 2)) * 100

        self.values = {
                'loss': loss,
                'accuracy': accuracy
            }
        return accuracy

