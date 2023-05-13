import argparse
from phytools.configs.config import Config
# from phytools.models.PDENet2.model import PDENet2 as Net
from phytools.models.ODENet.model import ODENet as Net


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