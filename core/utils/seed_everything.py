import torch
import random
import numpy as np


def set_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
