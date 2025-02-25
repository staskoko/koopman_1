# Import for data manipulation and file handling
import pandas as pd
import numpy as np

# Import PyTorch and its modules for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random as r

import help_func

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define file path for the CSV file
file_path = '/home/trarity/koopman_1/data/simulation_results.csv'
def DiscreteSpectrumExampleFn(x1range, x2range, numICs, tSpan, mu, lam, seed):
    """
    Generates trajectories based on a 3D Koopman linear system for a 2D nonlinear dynamical system.

    Args:
        x1range (tuple): Range (min, max) for x1 initial condition.
        x2range (tuple): Range (min, max) for x2 initial condition.
        numICs (int): Number of initial conditions.
        tSpan (array-like): 1D array of time points.
        mu (float): Parameter mu.
        lam (float): Parameter lambda.
        seed (int): Random seed for reproducibility.

    Returns:
        X (torch.Tensor): Trajectories of shape [numICs, len(tSpan), 2].
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Ensure tSpan is a tensor
    if not isinstance(tSpan, torch.Tensor):
        tSpan = torch.tensor(tSpan, dtype=torch.float32)

    lenT = tSpan.shape[0]

    # Generate random initial conditions for x1 and x2
    x1 = (x1range[1] - x1range[0]) * torch.rand(numICs, 1) + x1range[0]
    x2 = (x2range[1] - x2range[0]) * torch.rand(numICs, 1) + x2range[0]

    # Preallocate X with shape [numICs, lenT, 2]
    X = torch.zeros(numICs, lenT, 2, dtype=torch.float32)

    # Loop over each initial condition
    for j in range(numICs):
        # Construct initial state in the 3D Koopman space: [x1, x2, x1^2]
        # Note: x1[j] and x2[j] are 1-element tensors; we use item() to extract the value if needed,
        # but torch.tensor([...]) accepts tensors as elements.
        Y0 = torch.tensor([x1[j], x2[j], x1[j]**2], dtype=torch.float32)

        # Compute coefficients as in your MATLAB code
        c1 = Y0[0]
        c2 = Y0[1] + (lam * Y0[2]) / (2.0 * mu - lam)
        c3 = -(lam * Y0[2]) / (2.0 * mu - lam)
        c4 = Y0[2]

        # Compute the required exponentials over tSpan
        exp_mu_t = torch.exp(mu * tSpan)
        exp_lambda_t = torch.exp(lam * tSpan)
        exp_2mu_t = torch.exp(2.0 * mu * tSpan)

        # Construct the 3D trajectory Y with shape [3, lenT]
        Y = torch.vstack([
            c1 * exp_mu_t,
            c2 * exp_lambda_t + c3 * exp_2mu_t,
            c4 * exp_2mu_t
        ])

        # Extract the first two rows (x1 and x2 dynamics) and transpose to shape [lenT, 2]
        X[j, :, :] = Y[:2, :].T

    return X

def custom_loss(x_pred, x_target):
    total_loss = torch.sum(torch.mean((x_pred - x_target) ** 2))
    return total_loss

def loss_recon(xk, phi, phi_inv): # inputs (xk,encoder,decoder)
    pred = phi_inv(phi(xk[:, 0, :]))
    recon_loss = F.mse_loss(pred, xk[:, 0, :], reduction='mean')
    return recon_loss

def loss_pred(xk, S_p, K, phi, phi_inv): # inputs (xk, Sp = 30, Koopman, encoder, decoder)

    total_loss = torch.tensor(0.0, device=xk[:, 0, :].device)

    for m in range(1, S_p + 1):
        pred_loop = phi(xk[:, 0, :])
        for j in range(m):
            pred_loop = K(pred_loop)

        pred = phi_inv(pred_loop)
        total_loss += F.mse_loss(pred, xk[:, m, :], reduction='mean')

    pred_loss = total_loss / S_p
    #print(f"Predicted loss: {pred_loss}")

    return pred_loss

def loss_lin(xk, T, K, phi, phi_inv):# inputs (xk, Koopman, encoder, decoder)

    #T = len(xk[0, :, :])
    total_loss = torch.tensor(0.0, device=xk[:, 0, :].device)
    for m in range(1, T - 1):
        pred_loop = phi(xk[:, 0, :])
        for j in range(m):
          pred_loop = K(pred_loop)

        actual = phi(xk[:, m, :])

        total_loss += F.mse_loss(pred_loop, actual, reduction='mean')

    lin_loss = total_loss / (T - 1)
    #print(f"Linear loss: {lin_loss}")

    return lin_loss

def loss_inf(xk, K, phi, phi_inv):

    #first_term = loss_recon(xk, phi, phi_inv).abs().max()
    #second_term = loss_pred(xk, 1, K, phi, phi_inv).abs().max()

    x1 = xk[:, 0, :]
    recon_pred = phi_inv(phi(x1))
    first_term = (x1 - recon_pred).abs().max(dim=1)[0].mean()

    one_step_latent = K(phi(x1))
    one_step_pred = phi_inv(one_step_latent)
    second_term = (xk[:, 1, :] - one_step_pred).abs().max(dim=1)[0].mean()

    inf_loss = first_term + second_term
    #print(f"Inf loss: {inf_loss}")

    return inf_loss

def total_loss(alpha, W, xk, S_p, T, K, phi, phi_inv): # inputs (alpha, W, xk, Sp = 30, Koopman, encoder, decoder)
    alpha_1 = alpha[0]
    alpha_2 = alpha[1]
    alpha_3 = alpha[2]
    #W_norm_sqr = torch.sqtr(W * W)

    L_recon = loss_recon(xk, phi, phi_inv)
    L_pred = loss_pred(xk, S_p, K, phi, phi_inv)
    L_lin = loss_lin(xk, T, K, phi, phi_inv)
    L_inf = loss_inf(xk, K, phi, phi_inv)

    L_total = alpha_1*(L_recon + L_pred) + L_lin + alpha_2*L_inf # + alpha_3*W_norm_sqr
    #print(f"Total loss: {L_total}")

    return L_total

def self_feeding(model, xk):
    initial_input = xk[:, 0, :]
    num_steps = int(len(xk[0, :, 0]))

    predictions = []
    predictions.append(initial_input.detach())

    for step in range(num_steps - 1):
        x_pred = model(initial_input)
        predictions.append(x_pred.detach())
        initial_input = x_pred

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xk)
    return predictions, loss


# Data Generation

numICs = 7143
x1range = (-0.5, 0.5)
x2range = x1range
tSpan = torch.arange(0.0, 1.02, 0.02)  # equivalent to MATLAB: 0:0.02:1
mu = -0.05
lam = -1

# Create test, validation, and training tensors with different percentages of numICs
seed = 1
test_tensor = DiscreteSpectrumExampleFn(x1range, x2range, round(0.1 * numICs), tSpan, mu, lam, seed)
print("Test tensor shape:", test_tensor.shape)  # Expected: [0.1*numICs, len(tSpan), 2]

seed = 2
val_tensor = DiscreteSpectrumExampleFn(x1range, x2range, round(0.2 * numICs), tSpan, mu, lam, seed)
print("Validation tensor shape:", val_tensor.shape)  # Expected: [0.2*numICs, len(tSpan), 2]

seed = 3
train_tensor = DiscreteSpectrumExampleFn(x1range, x2range, round(0.7 * numICs), tSpan, mu, lam, seed)
print(f"Training tensor shape (seed {seed}):", train_tensor.shape)


# NN Structure

Num_meas = 2
Num_Obsv = 3
Num_Neurons = 30

class AUTOENCODER(nn.Module):
    def __init__(self, Num_meas, Num_Obsv, Num_Neurons):
        super(AUTOENCODER, self).__init__()

        # Neural network layers for the Encoder
        self.Encoder_In = nn.Linear(Num_meas, Num_Neurons, bias=True)
        self.Encoder_Hdd = nn.Linear(Num_Neurons, Num_Neurons, bias=True)
        self.Encoder_Hdd2 = nn.Linear(Num_Neurons, Num_Neurons, bias=True)
        self.Encoder_out = nn.Linear(Num_Neurons, Num_Obsv, bias=True)

        # Linear condition
        self.Koopman = nn.Linear(Num_Obsv, Num_Obsv, bias=False)

        # Neural network layers for the Decoder
        self.Decoder_In = nn.Linear(Num_Obsv, Num_Neurons, bias=True)
        self.Decoder_Hdd = nn.Linear(Num_Neurons, Num_Neurons, bias=True)
        self.Decoder_Hdd2 = nn.Linear(Num_Neurons, Num_Neurons, bias=True)
        self.Decoder_out = nn.Linear(Num_Neurons, Num_meas, bias=True)

        # Apply custom weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier Uniform initialization for weights
                torch.nn.init.xavier_uniform_(m.weight)*4
                # Zero initialization for biases
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def Encoder(self, x):
        x = F.relu(self.Encoder_In(x))
        x = F.relu(self.Encoder_Hdd(x))
        x = F.relu(self.Encoder_Hdd2(x))
        x = self.Encoder_out(x)
        return x

    def Koopman_op(self, x):
        x = self.Koopman(x)
        return x

    def Decoder(self, x):
        x = F.relu(self.Decoder_In(x))
        x = F.relu(self.Decoder_Hdd(x))
        x = F.relu(self.Decoder_Hdd2(x))
        x = self.Decoder_out(x)
        return x

    def forward(self, x_k):
        y_k = self.Encoder(x_k)
        y_k1 = self.Koopman_op(y_k)
        x_k1 = self.Decoder(y_k1)

        return x_k1

# Instantiate the model and move it to the GPU (if available)
model = AUTOENCODER(Num_meas, Num_Obsv, Num_Neurons).to(device)


# Training Loop

eps = 2000        # Number of epochs per batch size
lr = 1e-3        # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor[0, :, :])
alpha = [0.1, 10e-7, 10e-15]
W = 0
NN_structure = 'AUTOENCODER'

train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

M = 4  # Amount of models you want to run
Model_path = []
Models_loss_list = []
Running_Losses_Array = []

for i in range(M):
    Model_path.append(f"/home/trarity/koopman_1/Autoencoder_model_params{i}.pth")

for model_path_i in Model_path:
    training_attempt = 0
    while True:  # Re-run the training loop until no NaN is encountered
        training_attempt += 1
        print(f"\nStarting training attempt #{training_attempt} for model {model_path_i}")

        # Instantiate the model and optimizer afresh
        model = AUTOENCODER(Num_meas, Num_Obsv, Num_Neurons)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_list = []
        running_loss_list = []
        nan_found = False  # Flag to detect NaNs

        for e in range(eps):
            running_loss = 0.0
            for (batch_x,) in train_loader:
                optimizer.zero_grad()
                loss = total_loss(alpha, W, batch_x, S_p, T, model.Koopman_op, model.Encoder, model.Decoder)

                # Check if loss is NaN; if so, break out of loops
                if torch.isnan(loss):
                    nan_found = True
                    print(f"NaN detected at epoch {e+1}. Restarting training attempt.")
                    break

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            if nan_found:
                break

            avg_loss = running_loss / len(train_loader)
            loss_list.append(avg_loss)
            running_loss_list.append(running_loss)
            print(f'Epoch {e+1}, Avg Loss: {avg_loss:.10f}, Running loss: {running_loss:.3e}')
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate: {current_lr:.8f}')

            # Save the model parameters at the end of each epoch
            torch.save(model.state_dict(), model_path_i)

        # If no NaN was found during this training attempt, we exit the loop
        if not nan_found:
            break
        else:
            print("Restarting training loop due to NaN encountered.\n")

    Models_loss_list.append(running_loss)
    Running_Losses_Array.append(running_loss_list)
    torch.save(model.state_dict(), model_path_i)

    for (batch_x,) in test_loader:
      [traj_prediction, loss] = self_feeding(model, batch_x)
      running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)
    print(f'Test Data w/Model {c_m + 1}, Avg Loss: {avg_loss:.10f}, Running loss: {running_loss:.3e}')
    Test_loss_list.append(running_loss)
    c_m += 1
    
# Find the best of the models
Lowest_loss = min(Models_loss_list)
Lowest_test_loss = min(Test_loss_list)

Lowest_loss_index = Models_loss_list.index(Lowest_loss)
print(f"The best model has a running loss of {Lowest_loss} and is model nr. {Lowest_loss_index + 1}")
Lowest_test_loss_index = Test_loss_list.index(Lowest_test_loss)
print(f"The best model has a test running loss of {Lowest_test_loss} and is model nr. {Lowest_test_loss_index + 1}")

# Load the parameters of the best model
model.load_state_dict(torch.load(Model_path[Lowest_test_loss_index]))
print(f"Loaded model parameters from Model {Lowest_test_loss_index + 1}: {Model_path[Lowest_test_loss_index]}")


# Result Plotting

# Choose three distinct sample indices
sample_indices = r.sample(range(val_tensor.shape[0]), 3)
[Val_pred_traj, val_loss] = self_feeding(model, val_tensor)

print(f"Running loss for validation: {val_loss:.3e}")

fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, idx in enumerate(sample_indices):

    predicted_traj = Val_pred_traj[idx]
    actual_traj = val_tensor[idx]

    time_steps = range(val_tensor.shape[1])

    # Plot x1 in the first row
    axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
    axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
    axs[0, i].set_title(f"Validation Sample {idx} (x1)")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("x1")
    axs[0, i].legend()

    # Plot x2 in the second row
    axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x2')
    axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
    axs[1, i].set_title(f"Validation Sample {idx} (x2)")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("x2")
    axs[1, i].legend()

plt.tight_layout()
plt.show()

# Choose three distinct sample indices
sample_indices = r.sample(range(train_tensor.shape[0]), 3)
[train_pred_traj, train_loss] = self_feeding(model, train_tensor)

print(f"Running loss for training: {train_loss:.3e}")

fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, idx in enumerate(sample_indices):

    predicted_traj = train_pred_traj[idx]
    actual_traj = train_tensor[idx]

    time_steps = range(train_tensor.shape[1])

    # Plot x1 in the first row
    axs[0, i].plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
    axs[0, i].plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
    axs[0, i].set_title(f"Train Sample {idx} (x1)")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("x1")
    axs[0, i].legend()

    # Plot x2 in the second row
    axs[1, i].plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x2')
    axs[1, i].plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
    axs[1, i].set_title(f"Train Sample {idx} (x2)")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("x2")
    axs[1, i].legend()

plt.tight_layout()
plt.show()
