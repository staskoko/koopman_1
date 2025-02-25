import torch
import torch.nn.functional as F

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

