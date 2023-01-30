from got10k.experiments import *
from siamfc import TrackerSiamFC

import numpy as np
np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    net_path = "/home/snorlax/Projects/SiamFC-pytorch/siamfc_alexnet_e50.pth"
    # net_path = "/home/snorlax/Projects/SiamFC-pytorch/models/SiamFC_100.pth"
    data_dir = "/home/snorlax/datasets/track/OTB100"

    track = TrackerSiamFC(net_path)
    experiment = ExperimentOTB(root_dir=data_dir, version=2015)

    experiment.run(track)
    experiment.report([track.name])
