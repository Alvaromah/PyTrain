import pickle
import random
import numpy as np

import torch

def random_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
random_seed()

def R2(v): return "{:n}".format(v) if type(v) is int else "{:4.2f}".format(v)
def R3(v): return "{:n}".format(v) if type(v) is int else "{:5.3f}".format(v)
def R4(v): return "{:n}".format(v) if type(v) is int else "{:6.4f}".format(v)
def R5(v): return "{:n}".format(v) if type(v) is int else "{:7.5f}".format(v)
def R6(v): return "{:n}".format(v) if type(v) is int else "{:8.6f}".format(v)

def load_object(fn):
    with open(fn, 'rb') as file:
        return pickle.load(file)

def save_object(obj, fn):
    with open(fn, 'wb') as file:
        pickle.dump(obj, file)

def create_tensor(x, shape):
    if isinstance(x, list):
        x = np.array(x)
    x = x.reshape(shape).astype(np.float32)
    return torch.from_numpy(x)
