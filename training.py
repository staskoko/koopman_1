

# Training loop variables
eps = 200 # Number of epochs per batch size
lr = 1e-4 # Learning rate
batch_size = 256
S_p = 30
T = len(train_tensor[0, :, :])
alpha = [0.1, 10e-7, 10e-15]
W = 0
NN_structure = 'AUTOENCODER'

train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

M = 1 # Amount of models you want to run
Model_path = []
Models_loss_list = []
Running_Losses_Array = []

for i in range(M):
  Model_path.append(f"/content/drive/MyDrive/Colab Notebooks/Autoencoder_model_params{i}.pth")

for model_path_i in Model_path:

    # Instantiate the LNN model
    model = AUTOENCODER(Num_meas, Num_Obsv, Num_Neurons)
    loss_list = []
    running_loss_list = []
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(eps):
        running_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()

            loss = total_loss(alpha, W, batch_x, S_p, T, model.Koopman_op, model.Encoder, model.Decoder)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Clip gradients to prevent them from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        # Log the average loss per epoch
        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)
        running_loss_list.append(running_loss)
        print(f'Epoch {e + 1}, Loss: {avg_loss:.10f}, Running loss: {running_loss:.10f}')

        # Log the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.8f}')

        # Save the model parameters at the end of each epoch
        torch.save(model.state_dict(), model_path_i)

    Models_loss_list.append(running_loss)
    Running_Losses_Array.append(running_loss_list)
    torch.save(model.state_dict(), model_path_i)

# Find the best of the models
Lowest_loss = min(Models_loss_list)

# Find the index
Lowest_loss_index = Models_loss_list.index(Lowest_loss)
print(f"The best model has a loss of {Lowest_loss} and is model nr. {Lowest_loss_index}")

# Load the parameters of the best model
model.load_state_dict(torch.load(Model_path[Lowest_loss_index]))
#save_drive_best = f"/content/drive/MyDrive/Colab Notebooks/{NN_structure}_{running_loss}_Loss_Params.pth"
#torch.save(model.state_dict(), save_drive_best)

# Export running losses to Excel
running_loss_df = pd.DataFrame(Running_Losses_Array).transpose()
#running_loss_df.to_excel(f"/content/drive/MyDrive/Colab Notebooks/running_loss_{NN_structure}.xlsx", index=False)
