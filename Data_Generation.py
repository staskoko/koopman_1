import torch
import torch.nn.functional as F
import help_func

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

def DataGenerator(x1range, x2range, numICs, tSpan, mu, lam):
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
  
    return train_tensor, test_tensor, val_tensor
