import datetime

class BaseLogger():
    def __init__(self):
        pass

    def log(self, epoch, values, phase):
        if phase == 'train':
            print(epoch, 'train:', values, sep='\t', end='\t')
        elif phase == 'valid':
            print('  |', 'valid:', values, sep='\t')
        elif phase == 'eval':
            print(values, sep='\t')

