#---------------------------------------------------------------------------------------------------#
#preprocessing data
import random, numpy as np, pandas as pd

#---------------------------------------------------------------------------------------------------#
#train neuronal network
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class MyDataset(Dataset):
    def __init__(self, X, targets):
        self.X = torch.tensor(X, dtype=torch.float32)#torch.from_numpy(X) # Convert the NumPy array to a PyTorch tensor
        self.targets = torch.tensor(targets, dtype=torch.float32)#torch.from_numpy(targets) # Convert the NumPy array to a PyTorch tensor
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        x = self.X[index]
        y = self.targets[index]
        return x, y

class NeuralNetwork(nn.Module):
    def __init__(self,neurons):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        for i in range(len(neurons)-1):
            if i != 0: layers.append(nn.ReLU())
            layers.append(nn.Linear(neurons[i],neurons[i+1]))
        self.linear_relu_stack = nn.Sequential(*layers)
    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y

def train_nn(learning_rate:float,
             batch_size:int,
             layers:list,
             epochs:int,
             dataset:MyDataset,
             X_val:torch.tensor,
             Y_val:torch.tensor,
             input_size=8,
             output_size=200,
             group=None):

    config = {"learning_rate": learning_rate,
              "batch_size": batch_size,
              "epochs":epochs,
              "n_hidden_layers":len(layers),
              "data_size":len(dataset)}

    neurons = [input_size]
    for i,layer in enumerate(layers):
        config[f"hidden_layer_{i}"] = layer
        neurons.append(layer)
    neurons.append(output_size)

    # Initialize the model, loss function, and optimizer
    model = NeuralNetwork(neurons=neurons)
    criterion = torch.nn.MSELoss()#nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(model)
    
    try:
        # Training loop   
        for epoch in range(epochs):
            epoch_loss = 0 #average loss of this epoch
            validation_loss = 0
            for i, batch in enumerate(dataloader):
                inputs, targets = batch
                model.train() # Set the model to train mode
                optimizer.zero_grad() # Zero the gradients
                outputs = model(inputs) # Forward pass
                loss = criterion(outputs, targets) # Compute the loss
                loss.backward() # Backward pass
                optimizer.step() # Update the weights
                epoch_loss += loss.item()
            epoch_loss /= i
            model.eval() # Set the model to evaluation mode
            with torch.no_grad():
                # Forward pass for validation set
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, Y_val)
            print(f"Epoch {epoch:4.0f} | epoch_loss {epoch_loss:10.6f} | validation_loss {val_loss:10.6f}")
    except KeyboardInterrupt:
        pass
    return model, criterion, val_loss.item()#epoch_loss #used as metric for optimization
