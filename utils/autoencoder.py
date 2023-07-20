import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(Autoencoder, self).__init__()
        self.encoder = self._get_encoder(input_size, hidden_sizes)
        self.decoder = self._get_decoder(input_size, hidden_sizes)


    def _get_encoder(self, input_size, hidden_sizes):
        encoder_layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

            encoder_layers.append(nn.ReLU())
        return nn.Sequential(*encoder_layers)
    
    def _get_decoder(self, input_size, hidden_sizes):
        decoder_sizes = hidden_sizes[::-1]
        decoder_layers = []
        for i in range(len(decoder_sizes)-1):
            decoder_layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(decoder_sizes[-1], input_size))
        return nn.Sequential(*decoder_layers)
    
    def forward(self, X):
        encoded = self.encoder(X)
        return self.decoder(encoded)