import torch
import torch.nn.functional as F
import help_func

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
