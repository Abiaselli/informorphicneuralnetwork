import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

# ---------------- PID Approximation Utilities ----------------

def bin_inputs(X, n_bins=5):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    return est.fit_transform(X.detach().cpu().numpy())

def empirical_joint_probs(x_r, x_c, y):
    data = np.stack([x_r, x_c, y], axis=1)
    unique, counts = np.unique(data, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return unique, probs

def estimate_mutual_information(x, y):
    return mutual_info_score(x, y)

def estimate_pid_atoms(xR, xC, Y, bins=5):
    # Bin all variables
    xR_binned = bin_inputs(xR, bins)[:, 0]
    xC_binned = bin_inputs(xC, bins)[:, 0]
    y_binned = (Y.detach().cpu().numpy() > 0.5).astype(int)

    # Mutual Information Estimates
    I_Y_XR = estimate_mutual_information(y_binned, xR_binned)
    I_Y_XC = estimate_mutual_information(y_binned, xC_binned)
    I_Y_XR_XC = estimate_mutual_information(y_binned, xR_binned + xC_binned * bins)

    # Approximate PID atoms (heuristic decomposition)
    I_red = max(0, min(I_Y_XR, I_Y_XC))
    I_unq_R = max(0, I_Y_XR - I_red)
    I_unq_C = max(0, I_Y_XC - I_red)
    I_syn = max(0, I_Y_XR_XC - I_red - I_unq_R - I_unq_C)
    Hres = 1.0 - I_Y_XR_XC  # entropy of Y minus info explained

    return (
        torch.tensor(I_unq_R, dtype=torch.float32),
        torch.tensor(I_unq_C, dtype=torch.float32),
        torch.tensor(I_red, dtype=torch.float32),
        torch.tensor(I_syn, dtype=torch.float32),
        torch.tensor(Hres, dtype=torch.float32),
    )

# ---------------- Infomorphic Neuron ----------------

class InfomorphicNeuron(nn.Module):
    def __init__(self, input_dim_R, input_dim_C):
        super().__init__()
        self.wR = nn.Parameter(torch.randn(input_dim_R) * 0.01)
        self.wC = nn.Parameter(torch.randn(input_dim_C) * 0.01)
        self.bias_R = nn.Parameter(torch.zeros(1))
        self.bias_C = nn.Parameter(torch.zeros(1))

    def forward(self, xR, xC):
        r = F.linear(xR, self.wR, self.bias_R)
        c = F.linear(xC, self.wC, self.bias_C)

        # Biology-inspired nonlinear modulator
        A = r * (0.5 + torch.sigmoid(2 * r * c))
        prob = torch.sigmoid(A)
        return prob

    def compute_pid_loss(self, xR, xC, y, gamma, bins=5):
        Iunq_R, Iunq_C, Ired, Isyn, Hres = estimate_pid_atoms(xR, xC, y, bins)

        G = (gamma['unq_R'] * Iunq_R +
             gamma['unq_C'] * Iunq_C +
             gamma['red']   * Ired +
             gamma['syn']   * Isyn +
             gamma['res']   * Hres)
        return -G  # We minimize loss (negative of info gain)

# ---------------- Example Training Loop ----------------

def train_step(neuron, xR, xC, y, gamma, optimizer):
    neuron.train()
    optimizer.zero_grad()

    y_pred = neuron(xR, xC)
    loss = neuron.compute_pid_loss(xR, xC, y_pred, gamma)
    loss.backward()
    optimizer.step()
    return loss.item()

# ---------------- Demo ----------------

if __name__ == "__main__":
    input_dim_R = 5
    input_dim_C = 2
    neuron = InfomorphicNeuron(input_dim_R, input_dim_C)

    optimizer = torch.optim.Adam(neuron.parameters(), lr=1e-2)

    gamma = {
        'unq_R': 0.1,
        'unq_C': 0.1,
        'red':   1.0,
        'syn':   0.1,
        'res':   0.0
    }

    # Random dummy data (can replace with MNIST, etc.)
    for epoch in range(30):
        xR = torch.rand(32, input_dim_R)
        xC = torch.randint(0, 2, (32, input_dim_C)).float()
        y = torch.randint(0, 2, (32,)).float()

        loss = train_step(neuron, xR, xC, y, gamma, optimizer)
        print(f"Epoch {epoch + 1:2d} | Loss: {loss:.5f}")
