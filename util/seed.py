import torch
import random
import numpy as np



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    return generator
