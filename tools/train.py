import os.path
from siamfc import train_siamfc, GOT10k

def all_seed(seed=1):
    import torch
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    all_seed(seed=3407)

    root_dir = os.path.abspath('/media/snorlax/My Passport/datasets/GOT10k/')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    train_siamfc(seqs)

