import torch
import pandas as pd
import numpy as np

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