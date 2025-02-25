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
import nn_structure

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define file path for the CSV file
file_path = '/home/trarity/koopman_1/data/simulation_results.csv'

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
