import torch
import torch.nn.functional as F

def euler_step(f, x, h):
    x_new = x + h * f(x)
    return x_new

def train_euler(f, x, h):
    x_pred_1 = euler_step(f, x, h) #Predicts x dx p1 p2 at time t+1
    x_pred_2 = euler_step(f, x_pred_1, h) #Predicts x at time t+2
    x_pos_2 = x_pred_2[:, 0]
    x_pos_2 = x_pos_2.unsqueeze(1) #Go from size ([32,]) to ([32,1])
    x_pred = torch.cat((x_pos_2, x_pred_1[:, [1]]),  axis=1) #Outputs x at time t+2 and dx p1 p2 at time t+1
    return x_pred

def loss_recon(xk, phi, phi_inv): # inputs (xk,encoder,decoder)
    pred = phi_inv(phi(xk[:, 0, :]))
    recon_loss = F.mse_loss(pred, xk[:, 0, :], reduction='mean')
    return recon_loss

def loss_pred(xk, S_p, K, phi, phi_inv): # inputs (xk, Sp = 30, Koopman, encoder, decoder)

    total_loss = torch.tensor(0.0, device=xk[:, 0, :].device)
    pred_loop = phi(xk[:, 0, :])
    for m in range(1, S_p + 1):
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
    pred_loop = phi(xk[:, 0, :])
    for m in range(1, T - 1):
        for j in range(m):
          pred_loop = K(pred_loop)

        actual = phi(xk[:, m, :])

        total_loss += F.mse_loss(pred_loop, actual, reduction='mean')

    lin_loss = total_loss / (T - 1)
    #print(f"Linear loss: {lin_loss}")

    return lin_loss

def loss_inf(xk, K, phi, phi_inv):

    first_term = loss_recon(xk, phi, phi_inv).abs().max()
    second_term = loss_pred(xk, 1, K, phi, phi_inv).abs().max()

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