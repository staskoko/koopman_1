import torch
import torch.nn as nn
import torch.nn.functional as F

class AUTOENCODER(nn.Module):
    def __init__(self, Num_meas, Num_Obsv, Num_Neurons):
        super(AUTOENCODER, self).__init__()

        # Neural network layers for the Encoder
        self.Encoder_In = nn.Linear(Num_meas, Num_Neurons)
        self.Encoder_Hdd = nn.Linear(Num_Neurons, Num_Neurons)
        self.Encoder_out = nn.Linear(Num_Neurons, Num_Obsv)

        # Linear condition
        self.Koopman = nn.Linear(Num_Obsv, Num_Obsv, bias=False)

        # Neural network layers for the Decoder
        self.Decoder_In = nn.Linear(Num_Obsv, Num_Neurons)
        self.Decoder_Hdd = nn.Linear(Num_Neurons, Num_Neurons)
        self.Decoder_out = nn.Linear(Num_Neurons, Num_meas)

        # Apply custom weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier Uniform initialization for weights
                torch.nn.init.xavier_uniform_(m.weight)
                # Zero initialization for biases
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def Encoder(self, x):
        x = F.relu(self.Encoder_In(x))
        x = F.relu(self.Encoder_Hdd(x))
        x = F.relu(self.Encoder_out(x))
        return x

    def Koopman_op(self, x):
        x = self.Koopman(x)
        return x

    def Decoder(self, x):
        x = F.relu(self.Decoder_In(x))
        x = F.relu(self.Decoder_Hdd(x))
        x = F.relu(self.Decoder_out(x))
        return x

    def forward(self, x_k):

        y_k = self.Encoder(x_k)
        y_k1 = self.Koopman_op(y_k)
        x_k1 = self.Decoder(y_k1)

        return x_k1

model = AUTOENCODER(Num_meas, Num_Obsv, Num_Neurons)
