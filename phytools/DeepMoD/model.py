import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from deepymod import DeepMoD as deepmod
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from dataset.burgers import burgers_delta
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DeepMoD():
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):

        # Making dataset
        v = 0.1
        A = 1.0

        x = torch.linspace(-3, 4, 100)
        t = torch.linspace(0.5, 5.0, 50)
        load_kwargs = {"x": x, "t": t, "v": v, "A": A}
        preprocess_kwargs = {"noise_level": 0.05}

        dataset = Dataset(
            burgers_delta,
            load_kwargs=load_kwargs,
            preprocess_kwargs=preprocess_kwargs,
            subsampler=Subsample_random,
            subsampler_kwargs={"number_of_samples": 500},
            device=device,
        )

        # coords = dataset.get_coords().cpu()
        # data = dataset.get_data().cpu()
        # fig, ax = plt.subplots()
        # im = ax.scatter(coords[:,1], coords[:,0], c=data[:,0], marker="x", s=10)
        # ax.set_xlabel('x')
        # ax.set_ylabel('t')
        # fig.colorbar(mappable=im)
        #
        # plt.show()

        train_dataloader, test_dataloader = get_train_test_loader(
            dataset, train_test_split=0.8
        )

        network = NN(2, [50, 50, 50, 50], 1)

        library = Library1D(poly_order=2, diff_order=3)

        estimator = Threshold(0.1)
        sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5)

        constraint = LeastSquares()
        # Configuration of the sparsity scheduler

        model = deepmod(network, library, estimator, constraint).to(device)

        # Defining optimizer
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)

        train(
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            sparsity_scheduler,
            exp_ID="Test",
            write_iterations=25,
            max_iterations=100000,
            delta=1e-4,
            patience=200,
        )

        # model.sparsity_masks

        print(model.estimator_coeffs())
