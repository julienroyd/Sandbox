import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

def eval_plot_mnist(x, x_hat, x_new, save_path):
    n_imgs = x.shape[0]
    gridspec_kw = dict(wspace=.25, hspace=.25)
    fig, ax = plt.subplots(nrows=n_imgs, ncols=3, figsize=(3 * 1.5, n_imgs * 1.5), gridspec_kw=gridspec_kw)
    for i, j in itertools.product(range(n_imgs), range(3)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    def invert_transforms(flat_img):
        img = flat_img.view(28, 28)
        return img * 0.3081 + 0.1307

    for k in range(n_imgs * 3):
        i = k // 3
        j = k % 3
        ax[i, j].cla()

        if i == 0:
            if j == 0:
                ax[i, j].set_title("Generated")
            elif j == 1:
                ax[i, j].set_title("Original")
            elif j == 2:
                ax[i, j].set_title("Reconstructed")

        if j == 0:
            img = invert_transforms(x_new[i])
        elif j == 1:
            img = invert_transforms(x[i])
        elif j == 2:
            img = invert_transforms(x_hat[i])
        else:
            raise ValueError

        ax[i, j].imshow(img.data.numpy(), cmap='gray')

    plt.tight_layout()

    os.makedirs(str(save_path.parent), exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)

    return


def eval_plot_gmix(x, x_hat, x_new, save_path):
    n_recons = 10

    gridspec_kw = dict(wspace=.25, hspace=.25)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), gridspec_kw=gridspec_kw)

    # def invert_transforms(x):
    #     return x * 0.9985 + 5.

    def invert_transforms(x):
        return x

    x = invert_transforms(x)
    x_hat = invert_transforms(x_hat)
    x_new = invert_transforms(x_new)

    # Plot with fake data over real data

    ax[0].scatter(x[:, 0], x[:, 1], color="blue", label="real data", alpha=0.5)
    ax[0].scatter(x_new[:, 0], x_new[:, 1], color="red", label="fake data", alpha=0.5)

    ax[0].legend(loc='lower left', bbox_to_anchor=(0., 1.), fancybox=True, shadow=False, ncol=2)

    # Plot with reconstructed data over real data (corresponding points)

    cm = plt.cm.get_cmap("jet")

    for i in range(n_recons):
        color = np.array(cm(float(i) / (n_recons - 1)))

        ax[1].scatter(x[i, 0], x[i, 1], color=color, label="real data" if i == 0 else None, s=100., alpha=0.3)
        ax[1].scatter(x_hat[i, 0], x_hat[i, 1], color=color, label="fake data" if i == 0 else None, s=30.)

    ax[1].legend(loc='lower left', bbox_to_anchor=(0., 1.), fancybox=True, shadow=False, ncol=2)

    # Ax limits to have a fixed frame from update to update

    mini_1, mini_2 = torch.min(x[:,0]), torch.min(x[:, 1])
    maxi_1, maxi_2 = torch.max(x[:, 0]), torch.max(x[:, 1])
    half_width_1, half_width_2 = abs(maxi_1 - mini_1) / 2., abs(maxi_2 - mini_2) / 2.
    ax[0].set_xlim(mini_1 - half_width_1, maxi_1 + half_width_1)
    ax[0].set_ylim(mini_2 - half_width_2, maxi_2 + half_width_2)
    ax[1].set_xlim(mini_1 - half_width_1, maxi_1 + half_width_1)
    ax[1].set_ylim(mini_2 - half_width_2, maxi_2 + half_width_2)

    # Save the figure

    plt.tight_layout()
    os.makedirs(str(save_path.parent), exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
