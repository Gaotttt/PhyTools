import yaml


class Config:

    @staticmethod
    def fromfile(filename):
        with open(filename) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg
