import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6  # EPS for numerical stability

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)  # orthogonal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, output_act=nn.Identity, logger=None, name="Unnamed Network"):
        super().__init__()
        self.name = name
        # Instantiate layers
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0], bias=True), nn.ReLU()])  # First layer
        for i in range(len(hidden_dims) - 1):
            self.layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True), nn.ReLU()])
        self.layers.extend([nn.Linear(hidden_dims[-1], output_dim, bias=True), output_act()])  # Last layer

        # Initialise parameters
        self.apply(weights_init_)

        # Prints architecture
        if logger is not None:
            logger.info(f"\n{name}:\n{self}")
        else:
            print(f"\n{name}:\n{self}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class simple_AE_model(nn.Module):
    def __init__(self, x_dim, hidden_dims, z_dim, logger=None):
        """
        A simple AE that uses MLP based encoder and decoder.
        It uses tanh() activations at the end of the encoder to be able to sample from a uniform
        """
        super().__init__()
        self.z_dim = z_dim

        # Instantiate the models
        self.encoder = MLP(input_dim=x_dim, hidden_dims=hidden_dims, output_dim=z_dim, output_act=nn.Tanh,
                           logger=logger, name="Encoder")
        self.decoder = MLP(input_dim=z_dim, hidden_dims=hidden_dims[::-1], output_dim=x_dim, output_act=nn.Identity,
                           logger=logger, name="Decoder")

        # Instantiate the prior
        self.prior = torch.distributions.uniform.Uniform(
            low=torch.FloatTensor([-1.] * z_dim),
            high=torch.FloatTensor([1.] * z_dim)
        )

    def encode(self, x):
        return self.encoder(x), None, None

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n_samples):
        z = self.prior.sample(sample_shape=(n_samples,))
        return self.decoder(z)


class simple_VAE_model(nn.Module):
    def __init__(self, x_dim, hidden_dims, z_dim, logger=None):
        """
        A simple VAE that uses MLP based encoder and decoder.
        The prior is a standard multivariate gaussian.
        """
        super().__init__()
        self.z_dim = z_dim

        # Instantiate the models
        self.encoder = MLP(input_dim=x_dim, hidden_dims=hidden_dims, output_dim=2*z_dim, output_act=nn.Identity, logger=logger, name="Encoder")
        self.decoder = MLP(input_dim=z_dim, hidden_dims=hidden_dims[::-1], output_dim=x_dim, output_act=nn.Identity, logger=logger, name="Decoder")

        # Instantiate the prior
        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.FloatTensor([0.] * z_dim),
            covariance_matrix=torch.diag(torch.FloatTensor([1.] * z_dim)))

    def encode(self, x):
        # Forwards the datapoint x and get parameters mu and sigma for q(z|x)
        gaussian_params = self.encoder(x)
        posterior_mu = gaussian_params[:, :self.z_dim]
        posterior_std = torch.exp(gaussian_params[:, self.z_dim:])

        # Reparameterisation trick
        batch_size = x.shape[0]
        epsilon = self.prior.sample(sample_shape=(batch_size,))
        z = posterior_std * epsilon + posterior_mu

        return z, posterior_mu, posterior_std

    def decode(self, z):
        # Returns a set generated datapoints for a set of given z's (at training time)
        return self.decoder(z)

    def sample(self, n_samples):
        # Returns set of generated datapoints by sampling the z's from the prior (at test time)
        z = self.prior.sample(sample_shape=(n_samples,))
        return self.decoder(z)
