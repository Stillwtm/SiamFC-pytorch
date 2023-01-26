from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    net_path = ""
    data_dir = "/home/snorlax/datasets/track/OTB100"

    track = TrackerSiamFC(net_path)
    experiment = ExperimentOTB(root_dir=data_dir, version=2015)

    experiment.run(track)
    experiment.report([track.name])
