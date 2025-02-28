import torch
import torch.nn.functional as F

def custom_loss(x_pred, x_target):
    total_loss = torch.sum(torch.mean((x_pred - x_target) ** 2))
    return total_loss

def loss_recon(xk, phi, phi_inv): # inputs (xk,encoder,decoder)
    pred = phi_inv(phi(xk[:, 0, :]))
    recon_loss = F.mse_loss(pred, xk[:, 0, :], reduction='mean')
    return recon_loss

def loss_pred(xk, S_p, K, phi, phi_inv): # inputs (xk, Sp = 30, Koopman, encoder, decoder)

    total_loss = torch.tensor(0.0, device=xk[:, 0, :].device)
    pred_loop = phi(xk[:, 0, :])
    for m in range(1, S_p + 1):
        pred_loop = K(pred_loop)
        pred = phi_inv(pred_loop)
        total_loss += F.mse_loss(pred, xk[:, m, :], reduction='mean')

    pred_loss = total_loss / S_p
    #print(f"Predicted loss: {pred_loss}")

    return pred_loss

def loss_lin(xk, T, K, phi, phi_inv):# inputs (xk, Koopman, encoder, decoder)
    #T = len(xk[0, :, :])
    total_loss = torch.tensor(0.0, device=xk[:, 0, :].device)
    pred_loop = phi(xk[:, 0, :])

    for m in range(1, T - 1):
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

    return L_total, L_recon, L_pred, L_lin, L_inf

