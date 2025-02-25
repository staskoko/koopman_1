import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from help_func import custom_loss, loss_recon, loss_pred, loss_lin, loss_inf, total_loss, self_feeding

def tensor_prep(file_path):
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
    val_ids = simulation_ids[10_000:15_000]
    test_ids = simulation_ids[15_000:]

    # Function to extract a (101, 2) sequence for a given simulation_id
    def extract_sequence_for_sim(sim_df):
        # Sort by the step index to ensure correct order
        sim_df = sim_df.sort_values('step')
        # Extract the x1 and x2 columns as a NumPy array (expected shape: [101, 2])
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
        # Stack along a new axis so that the resulting shape is (101, num_simulations, 2)
        final_array = np.stack(sequences, axis=1)
        return torch.tensor(final_array, dtype=torch.float32)

    # Generate the final 3D tensors for train, validation, and test sets
    train_tensor = extract_sequences_3d(df, train_ids)
    train_tensor = train_tensor.transpose(0, 1)
    val_tensor = extract_sequences_3d(df, val_ids)
    val_tensor = val_tensor.transpose(0, 1)
    test_tensor = extract_sequences_3d(df, test_ids)
    test_tensor = test_tensor.transpose(0, 1)

    return train_tensor, val_tensor, test_tensor

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

def DataGenerator(x1range, x2range, numICs, tSpan, mu, lam, file_path, type):
    if type == 0:
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
    
    else:
      train_tensor, val_tensor, test_tensor = tensor_prep(file_path)

    return train_tensor, test_tensor, val_tensor
