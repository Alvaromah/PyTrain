import datetime

class BaseLogger():
    def __init__(self):
        pass

    def log(self, epoch, phase, values):
        ts = F'{datetime.datetime.now().time()}'[:8]
        if phase == 'train':
            print(epoch, ts, 'train:', values, sep='\t', end='\t')
        elif phase == 'valid':
            print('|', 'valid:', values, sep='\t')

