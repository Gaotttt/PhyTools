import argparse
from phytools.config import Config
# from phytools.PDENet2.model import PDENet2 as Net
# from phytools.PDENet.model import PDENet as Net
# from phytools.ODENet.model import ODENet as Net
# from phytools.ODENet.model_mnist import ODENet as Net
# from phytools.DeepMoD.model import DeepMoD as Net
from phytools.MeshGraphNets.model import MeshGraphNets as Net

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Network')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', default='work_dir/', help='the dir to save logs and models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = Net(cfg)
    model.train()


if __name__ == '__main__':
    main()
