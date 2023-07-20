from utils.utility import load_data_trn, load_dae_config, to_cuda
from utils.autoencoder import Autoencoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_autoencoder(X_trn: torch.Tensor, dae_config: dict) -> list:
    batch_size              = dae_config["batch_size"]
    num_epochs_autoencoder  = 30
    learning_rate           = dae_config["alpha"]
    dataset_autoencoder     = TensorDataset(X_trn)
    dataloader_autoencoder  = DataLoader(dataset_autoencoder, batch_size=batch_size, shuffle=True)
    input_size              = X_trn.shape[1]
    hidden_sizes            = dae_config["encoders"].astype(int)
    autoencoder             = to_cuda(Autoencoder(input_size, hidden_sizes))
    criterion_autoencoder   = nn.MSELoss()
    optimizer_autoencoder   = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs_autoencoder):
        total_loss = 0.0
        for batch_data in dataloader_autoencoder:
            inputs_autoencoder  = batch_data[0]
            outputs_autoencoder = autoencoder(inputs_autoencoder)
            loss_autoencoder    = criterion_autoencoder(outputs_autoencoder, inputs_autoencoder)
            optimizer_autoencoder.zero_grad()
            loss_autoencoder.backward()
            optimizer_autoencoder.step()
            total_loss += loss_autoencoder.item()

        avg_loss = total_loss / len(dataloader_autoencoder)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs_autoencoder}], Autoencoder Loss: {avg_loss:.4f}")

    encoder_weights = []
    for layer in autoencoder.encoder:
        if isinstance(layer, nn.Linear):
            encoder_weights.append(to_cuda(layer.weight.detach().clone()))

    A = X_trn.clone()
    for weights in encoder_weights:
        A = torch.matmul(A, weights.t())

    return A, encoder_weights

def train_softmax(A: torch.Tensor, Y_trn: torch.Tensor, dae_config: dict) -> torch.Tensor:
    batch_size          = dae_config["batch_size"]
    num_epochs_softmax  = 50
    learning_rate       = dae_config["alpha"]
    dataset_encoded     = TensorDataset(A, Y_trn)
    dataloader_softmax  = DataLoader(dataset_encoded, batch_size=batch_size, shuffle=True)
    A                   = to_cuda(A)
    hidden_sizes        = dae_config["encoders"].astype(int)
    model_softmax       = to_cuda(nn.Linear(hidden_sizes[-1], Y_trn.shape[1]))
    criterion_softmax   = nn.CrossEntropyLoss()
    optimizer_softmax   = optim.Adam(model_softmax.parameters(), lr=learning_rate)

    for epoch in range(num_epochs_softmax):
        total_loss = 0.0
        for batch_data, batch_targets in dataloader_softmax:
            outputs_softmax = model_softmax(batch_data)
            loss_softmax    = criterion_softmax(outputs_softmax, torch.argmax(batch_targets, dim=1))
            optimizer_softmax.zero_grad()
            loss_softmax.backward()
            optimizer_softmax.step()
            total_loss += loss_softmax.item()

        avg_loss = total_loss / len(dataloader_softmax)
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs_softmax}], Softmax Loss: {avg_loss:.4f}")

    return model_softmax.weight.data.detach().clone()
