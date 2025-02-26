import torch
import torch.nn as nn
import torch.nn.functional as F

class AUTOENCODER(nn.Module):
    def __init__(self, Num_meas, Num_Obsv, Num_Neurons, Num_hidden_encoder, Num_hidden_decoder):
        super(AUTOENCODER, self).__init__()

        self.encoder_in = nn.Linear(Num_meas, Num_Neurons, bias=True)
        self.encoder_hidden = nn.ModuleList([nn.Linear(Num_Neurons, Num_Neurons, bias=True) for _ in range(Num_hidden_encoder)])
        self.encoder_out = nn.Linear(Num_Neurons, Num_Obsv, bias=True)

        self.Koopman = nn.Linear(Num_Obsv, Num_Obsv, bias=False)

        self.decoder_in = nn.Linear(Num_Obsv, Num_Neurons, bias=True)
        self.decoder_hidden = nn.ModuleList([nn.Linear(Num_Neurons, Num_Neurons, bias=True) for _ in range(Num_hidden_decoder)])
        self.decoder_out = nn.Linear(Num_Neurons, Num_meas, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)*4
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def Encoder(self, x):
        x = F.relu(self.encoder_in(x))
        for layer in self.encoder_hidden:
            x = F.relu(layer(x))
        x = self.encoder_out(x)
        return x

    def Koopman_op(self, x):
        return self.Koopman(x)

    def Decoder(self, x):
        x = F.relu(self.decoder_in(x))
        for layer in self.decoder_hidden:
            x = F.relu(layer(x))
        x = self.decoder_out(x)
        return x

    def forward(self, x_k):
        y_k = self.Encoder(x_k)
        y_k1 = self.Koopman_op(y_k)
        x_k1 = self.Decoder(y_k1)
        return x_k1

