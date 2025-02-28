import torch
import torch.nn.functional as F

from loss_func import custom_loss

def self_feeding(xk, K, phi, phi_inv):
    num_steps = int(len(xk[0, :, 0]))
    predictions = []
    predictions.append(xk[:, 0, :])
    x_k = xk[:, 0, :]
    for step in range(num_steps - 1):
        y_k = phi(x_k)
        y_k1 = K(y_k)
        x_pred = phi_inv(y_k1)
        predictions.append(x_pred)
        x_k = x_pred
    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xk)
    return predictions, loss

