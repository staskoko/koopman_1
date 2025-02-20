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

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define file path for the CSV file
file_path = '/home/trarity/koopman_1/data/simulation_results.csv'

# Load the CSV file into a DataFrame and sort by simulation_id and time
df = pd.read_csv(file_path)
df = df.sort_values(by=['simulation_id', 'time'])

# Create a step index within each simulation_id (assumes rows are ordered by time)
df['step'] = df.groupby('simulation_id').cumcount()

# Extract unique simulation IDs and shuffle them
simulation_ids = df['simulation_id'].unique()
np.random.shuffle(simulation_ids)

# Split simulation IDs into train (10,000), validation (5,000), and test (5,000)
train_ids = simulation_ids[:10_000]
val_ids   = simulation_ids[10_000:15_000]
test_ids  = simulation_ids[15_000:]

# Function to extract a (101, 2) sequence for a given simulation_id
def extract_sequence_for_sim(sim_df):
    sim_df = sim_df.sort_values('step')
    arr = sim_df[['x1', 'x2']].to_numpy()
    if arr.shape[0] != 101:
        print(f"Warning: simulation_id {sim_df['simulation_id'].iloc[0]} has {arr.shape[0]} rows instead of 101")
    return arr

# Function to process a list of simulation_ids into a 3D tensor with shape [101, num_simulations, 2]
def extract_sequences_3d(df, sim_ids):
    sequences = []
    for sim_id in sim_ids:
        sim_df = df[df['simulation_id'] == sim_id]
        seq = extract_sequence_for_sim(sim_df)  # Expected shape: (101, 2)
        sequences.append(seq)
    final_array = np.stack(sequences, axis=1)
    return torch.tensor(final_array, dtype=torch.float32)

# Generate the final 3D tensors for train, validation, and test sets
train_tensor = extract_sequences_3d(df, train_ids)
train_tensor = train_tensor.transpose(0, 1)
val_tensor   = extract_sequences_3d(df, val_ids)
val_tensor   = val_tensor.transpose(0, 1)
test_tensor  = extract_sequences_3d(df, test_ids)
test_tensor  = test_tensor.transpose(0, 1)

# Define the time step for integration
h = 0.001

def euler_step(f, x, h):
    return x + h * f(x)

def train_euler(f, x, h):
    x_pred_1 = euler_step(f, x, h)
    x_pred_2 = euler_step(f, x_pred_1, h)
    x_pos_2 = x_pred_2[:, 0].unsqueeze(1)
    x_pred = torch.cat((x_pos_2, x_pred_1[:, [1]]), axis=1)
    return x_pred

def loss_recon(xk, phi, phi_inv):
    pred = phi_inv(phi(xk[:, 0, :]))
    return F.mse_loss(pred, xk[:, 0, :], reduction='mean')

def loss_pred(xk, S_p, K, phi, phi_inv):
    total_loss = torch.tensor(0.0, device=xk[:, 0, :].device)
    pred_loop = phi(xk[:, 0, :])
    for m in range(1, S_p + 1):
        for j in range(m):
            pred_loop = K(pred_loop)
        pred = phi_inv(pred_loop)
        total_loss += F.mse_loss(pred, xk[:, m, :], reduction='mean')
    return total_loss / S_p

def loss_lin(xk, T, K, phi, phi_inv):
    total_loss = torch.tensor(0.0, device=xk[:, 0, :].device)
    pred_loop = phi(xk[:, 0, :])
    for m in range(1, T - 1):
        for j in range(m):
            pred_loop = K(pred_loop)
        actual = phi(xk[:, m, :])
        total_loss += F.mse_loss(pred_loop, actual, reduction='mean')
    return total_loss / (T - 1)

def loss_inf(xk, K, phi, phi_inv):
    first_term = loss_recon(xk, phi, phi_inv).abs().max()
    second_term = loss_pred(xk, 1, K, phi, phi_inv).abs().max()
    return first_term + second_term

def total_loss(alpha, W, xk, S_p, T, K, phi, phi_inv):
    alpha_1, alpha_2, alpha_3 = alpha
    L_recon = loss_recon(xk, phi, phi_inv)
    L_pred = loss_pred(xk, S_p, K, phi, phi_inv)
    L_lin = loss_lin(xk, T, K, phi, phi_inv)
    L_inf = loss_inf(xk, K, phi, phi_inv)
    return alpha_1*(L_recon + L_pred) + L_lin + alpha_2*L_inf

Num_meas = 2
Num_Obsv = 3
Num_Neurons = 30

class AUTOENCODER(nn.Module):
    def __init__(self, Num_meas, Num_Obsv, Num_Neurons):
        super(AUTOENCODER, self).__init__()
        self.Encoder_In = nn.Linear(Num_meas, Num_Neurons)
        self.Encoder_Hdd = nn.Linear(Num_Neurons, Num_Neurons)
        self.Encoder_Hdd2 = nn.Linear(Num_Neurons, Num_Neurons)
        self.Encoder_out = nn.Linear(Num_Neurons, Num_Obsv)
        self.Koopman = nn.Linear(Num_Obsv, Num_Obsv, bias=False)
        self.Decoder_In = nn.Linear(Num_Obsv, Num_Neurons)
        self.Decoder_Hdd = nn.Linear(Num_Neurons, Num_Neurons)
        self.Decoder_Hdd2 = nn.Linear(Num_Neurons, Num_Neurons)
        self.Decoder_out = nn.Linear(Num_Neurons, Num_meas)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def Encoder(self, x):
        x = F.relu(self.Encoder_In(x))
        x = F.relu(self.Encoder_Hdd(x))
        x = F.relu(self.Encoder_Hdd2(x))
        return F.relu(self.Encoder_out(x))

    def Koopman_op(self, x):
        return self.Koopman(x)

    def Decoder(self, x):
        x = F.relu(self.Decoder_In(x))
        x = F.relu(self.Decoder_Hdd(x))
        x = F.relu(self.Decoder_Hdd2(x))
        return F.relu(self.Decoder_out(x))

    def forward(self, x_k):
        y_k = self.Encoder(x_k)
        y_k1 = self.Koopman_op(y_k)
        return self.Decoder(y_k1)

# Instantiate the model and move it to the GPU (if available)
model = AUTOENCODER(Num_meas, Num_Obsv, Num_Neurons).to(device)

# Training loop variables
eps = 1600         # Number of epochs per batch size
lr = 1e-3         # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor[0, :, :])
alpha = [0.1, 10e-7, 10e-15]
W = 0
NN_structure = 'AUTOENCODER'

train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

M = 1  # Amount of models you want to run
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
          print(f'Epoch {e+1}, Loss: {avg_loss:.10f}, Running loss: {running_loss:.10f}')
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

# Find the best of the models
Lowest_loss = min(Models_loss_list)
Lowest_loss_index = Models_loss_list.index(Lowest_loss)
print(f"The best model has a loss of {Lowest_loss} and is model nr. {Lowest_loss_index}")

# Load the parameters of the best model (automatically on GPU if loaded into a model on GPU)
model.load_state_dict(torch.load(Model_path[Lowest_loss_index]))
running_loss_df = pd.DataFrame(Running_Losses_Array).transpose()

def predict_trajectory_full(initial_condition, steps, model):
    # Ensure the initial condition is on the same device as the model
    initial_condition = initial_condition.to(next(model.parameters()).device)
    predictions = [initial_condition]  # list to store each predicted state
    current_state = initial_condition

    for _ in range(steps):
        next_state = model(current_state)
        predictions.append(next_state)
        current_state = next_state

    trajectory = torch.cat(predictions, dim=0)
    return trajectory

for i in range(5):
    # Example usage:
    sample_index = r.randint(0, 4999) # choose the sample index
    actual_traj = test_tensor[sample_index]  # shape: [101, 2]
    initial_condition = actual_traj[0].unsqueeze(0)  # shape: [1,2]
    
    predicted_traj = predict_trajectory_full(initial_condition, steps=100, model=model)
    
    time_steps = range(101)
    # Font size adjustments for readability
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10
    
    # Plot for Variable 1
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, actual_traj[:, 0].cpu().numpy(), 'o-', label='True x1')
    plt.plot(time_steps, predicted_traj[:, 0].detach().cpu().numpy(), 'x--', label='Predicted x1')
    plt.xlabel("Time step")
    plt.ylabel("x1")
    plt.title("Comparison: True vs. Predicted Trajectory (x1)")
    plt.legend()
    plt.show()
    
    # Plot for Variable 2
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, actual_traj[:, 1].cpu().numpy(), 'o-', label='True x 2')
    plt.plot(time_steps, predicted_traj[:, 1].detach().cpu().numpy(), 'x--', label='Predicted x2')
    plt.xlabel("Time step")
    plt.ylabel("x2")
    plt.title("Comparison: True vs. Predicted Trajectory (x2)")
    plt.legend()
    plt.show()
