"""
The goal of this script is to implement a simple example of a regularised PINN for a differential equation.
The problem to solve is a 1D regression task on the sinus function.
We will compare two models, a regular MLP trained on the samples from sin(x)
and a regularised PINN trained on the same samples and the differential equation.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

N = 100
X_DATA_MIN = 0.
X_DATA_MAX = 4. * np.pi
X_GEN_MIN = 0.
X_GEN_MAX = 8. * np.pi
N_EPOCHS = 10000
LR = 1e-3

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden, activ=nn.ReLU, output_activ=nn.Tanh):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim), activ()])
        for _ in range(n_hidden):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), activ()])
        self.layers.extend([nn.Linear(hidden_dim, output_dim), output_activ()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train(x, y, model, optimiser, epochs=1000, use_pinn_regularisation=True):
    if use_pinn_regularisation:
        x.requires_grad = True
    criterion = nn.MSELoss(reduction='mean')
    for epoch in range(epochs):
        optimiser.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if use_pinn_regularisation:
            # Compute the gradients of the model with respect to the inputs
            grad_inputs = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
            pinn_loss_firstOrder = (grad_inputs - torch.cos(x)).pow(2).mean()
            loss += pinn_loss_firstOrder

            # Compute the second order derivative
            y_pred_2nd = model(grad_inputs)
            grad_secondOrder = torch.autograd.grad(y_pred_2nd, x, grad_outputs=torch.ones_like(y_pred_2nd), create_graph=True)[0]
            pinn_loss_secondOrder = (grad_secondOrder - torch.sin(x)).pow(2).mean()
            loss += pinn_loss_secondOrder
        
        loss.backward()
        optimiser.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model


if __name__ == "__main__":

    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Define the problem
    x = np.random.uniform(X_DATA_MIN, X_DATA_MAX, N)
    y = np.sin(x)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    # Define the model
    mlp = MLP(input_dim=1, output_dim=1, hidden_dim=64, n_hidden=3)
    mlp_pinn = MLP(input_dim=1, output_dim=1, hidden_dim=64, n_hidden=3)

    # Train the model
    mlp = train(torch.tensor(x).float(), torch.tensor(y).float(), mlp, optimiser=optim.Adam(mlp.parameters(), lr=LR), epochs=N_EPOCHS, use_pinn_regularisation=False)
    mlp_pinn = train(torch.tensor(x).float(), torch.tensor(y).float(), mlp_pinn, optimiser=optim.Adam(mlp_pinn.parameters(), lr=LR), epochs=N_EPOCHS, use_pinn_regularisation=True)

    # Plot the data and the real function
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    t = np.linspace(X_GEN_MIN, X_GEN_MAX, num=1000)
    ax.plot(t, np.sin(t), label='sin(x)', color='black', alpha=0.2, linewidth=20.)
    ax.scatter(x, y, color='red', label='samples')
    
    # Plot the MLP predictions
    y_pred = mlp(torch.tensor(t).float().reshape(-1, 1)).detach().numpy()
    ax.plot(t, y_pred, label='MLP', color='blue', linewidth=2.)

    # Plot the PINN predictions
    y_pred_pinn = mlp_pinn(torch.tensor(t).float().reshape(-1, 1)).detach().numpy()
    ax.plot(t, y_pred_pinn, label='PINN', color='orange', linewidth=2.)
    
    ax.legend()
    ax.grid()
    fig.savefig(Path(__file__).parent / 'pinn.png')