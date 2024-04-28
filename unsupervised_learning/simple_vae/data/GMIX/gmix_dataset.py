import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

import seaborn
seaborn.set_theme()

from alfred.utils.misc import uniquify

class GMIXdataset(torch.utils.data.Dataset):
    def __init__(self, path="."):
        """
        Loads a dataset of points sampled from a mixture of 2D gaussians
        """
        super().__init__()
        self.classes = None
        self.targets = None
        self.raw_folder = path

        self.data = torch.FloatTensor(np.load(path))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        y = torch.tensor([0.])
        return x, y

    @staticmethod
    def plot(data, path):
        fig, ax = plt.subplots()
        ax.scatter(data[:,0], data[:, 1])
        fig.savefig(path)
        plt.close(fig)


    @staticmethod
    def generate_dataset(n_points, gmm_means, gmm_covs, gmm_weights):

        # Draws the number of samples to be drawed from each component
        n_points_per_component = np.random.multinomial(n_points, pvals=gmm_weights)

        # Draws from each gaussian component
        pointer = 0
        data = np.empty(shape=(n_points, 2), dtype=np.float)

        for mean, cov, n in zip(gmm_means, gmm_covs, n_points_per_component):
            data[pointer:pointer+n] = np.random.multivariate_normal(np.array(mean), np.array(cov).reshape((2,2)), size=n)
            pointer += n

        # Save dataset and graph
        path = uniquify(Path("data.npy"))
        np.save(path, data)
        GMIXdataset.plot(data=data, path=f"{path.parent / path.stem}.png")

        return data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_points', type=int, default=500,
                        help="Number of points to be contained in the dataset.")

    parser.add_argument('-m', '--gmm_means', type=float, nargs='+', action='append', required=True,
                        help="Means of the mixture components. Should be a list of tuples with 2 floats each (2D gaussians), e.g. '-m 5 5 -m 0 0'.")

    parser.add_argument('-c', '--gmm_covs', type=float, nargs='+', action='append', required=True,
                        help="Covariance matrices of the mixture components. Should be a list of tuples with 4 floats each (2D gaussians), e.g. '-m 1 0 0 1 -m 1 0.5 0.5 1'.")

    parser.add_argument('-w', "--gmm_weights", nargs='+', required=True,
                        help="Weights of the mixture components. Should be a list of floats that sum to 1.")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    GMIXdataset.generate_dataset(args.n_points, args.gmm_means, args.gmm_covs, args.gmm_weights)
