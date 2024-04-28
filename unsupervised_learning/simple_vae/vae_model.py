import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from models import simple_VAE_model

from alfred.utils.plots import create_fig, plot_curves
from alfred.utils.recorder import remove_nones


class VAE(object):
    def __init__(self, x_dim, config, logger):

        self.device = torch.device(config.device)

        self.model = simple_VAE_model(x_dim=x_dim, hidden_dims=config.hidden_dims, z_dim=config.z_dim, logger=logger).to(device=self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=config.lr_decay_per_update)

        self.beta = config.beta

        self.metrics_to_record = {
            'update_i',
            'eval_update_i',
            'eval_epoch_i',
            'epoch_i',
            'reconstruction_loss',
            'prior_loss',
            'total_loss',
            'wallclock_time',
            'epoch_time',
            'lr'
        }

    def update_parameters(self, batch_data, update_i):

        x_batch = torch.FloatTensor(batch_data).to(self.device)

        z_batch, mu_batch, std_batch = self.model.encode(x=batch_data)
        x_hat_batch = self.model.decode(z_batch)

        # Reconstruction loss (MSE between original and reconstructed sample)
        reconstruction_loss = F.mse_loss(input=x_hat_batch, target=x_batch, reduction='none').sum(dim=1).mean(dim=0)

        # Prior loss (KL-divergence between posterior and prior distributions)
        # see: https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions
        prior_loss = 0.5 * ((mu_batch.square().sum(dim=1) + std_batch.square().sum(dim=1) - z_batch.shape[1] - torch.log(std_batch.square().prod(1)))).mean(dim=0)

        # Taking a gradient step
        total_loss = reconstruction_loss + self.beta * prior_loss

        self.model.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Bookkeeping

        new_recordings = {
            "update_i": update_i,
            "reconstruction_loss": reconstruction_loss.item(),
            "prior_loss": prior_loss.item(),
            "total_loss": total_loss.item(),
            "lr": self.scheduler.get_last_lr()
        }
        return new_recordings

    # Save model parameters
    def save_model(self, path, logger):
        self.to(torch.device("cpu"))
        logger.info(f'Saving models to {path}')
        torch.save(self.model.state_dict(), str(path / "vae_model.pt"))
        self.to(torch.device(self.device))

    # Load model parameters
    def load_model(self, path, logger):
        logger.info(f'Loading models from {path} and {path}')
        self.model.load_state_dict(torch.load(path / "vae_model.pt"))
        self.to(self.device)

    # Send models to different device
    def to(self, device):
        self.model.to(device)

    # Pytorch modules that wandb can monitor
    def wandb_watchable(self):
        return [self.model]

    # Graphs
    def create_plots(self, train_recorder, save_dir):

        fig, axes = create_fig((3, 3))
        plot_curves(axes[0, 0],
                    xs=[remove_nones(train_recorder.tape['update_i'])],
                    ys=[remove_nones(train_recorder.tape['reconstruction_loss'])],
                    xlabel='update_i',
                    ylabel='reconstruction_loss')
        plot_curves(axes[0, 1],
                    xs=[remove_nones(train_recorder.tape['update_i'])],
                    ys=[remove_nones(train_recorder.tape['prior_loss'])],
                    xlabel='update_i',
                    ylabel='prior_loss')
        plot_curves(axes[0, 2],
                    xs=[remove_nones(train_recorder.tape['update_i'])],
                    ys=[remove_nones(train_recorder.tape['total_loss'])],
                    xlabel='update_i',
                    ylabel='total_loss')
        plot_curves(axes[1, 0],
                    xs=[remove_nones(train_recorder.tape['update_i'])],
                    ys=[remove_nones(train_recorder.tape['lr'])],
                    xlabel='update_i',
                    ylabel='lr')

        plt.tight_layout()

        fig.savefig(str(save_dir / 'graphs.png'))
        plt.close(fig)
