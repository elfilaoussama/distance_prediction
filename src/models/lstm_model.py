# -*- coding: utf-8 -*-
"""
Z-Location Estimator Model for Deployment
Created on Mon May 23 04:55:50 2022
@author: ODD_team
Edited by our team : Sat Oct 4  11:00 PM 2024
@based on LSTM model 
"""

import torch
import torch.nn as nn
from config import CONFIG

device = CONFIG['device']

# Define the LSTM-based Z-location estimator model
class Zloc_Estimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(Zloc_Estimator, self).__init__()
        
        # LSTM layer
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        
        # Fully connected layers
        layersize = [306, 154, 76]
        layerlist = []
        n_in = hidden_dim
        for i in layersize:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU())
            n_in = i
        layerlist.append(nn.Linear(layersize[-1], 1))  # Final output layer
        
        self.fc = nn.Sequential(*layerlist)

    def forward(self, x):
        out, hn = self.rnn(x)
        output = self.fc(out[:, -1])  # Get the last output for prediction
        return output

# Deployment-ready class for handling the model
class LSTM_Model:
    def __init__(self):
        """
        Initializes the LSTM model for deployment with predefined parameters
        and loads the pre-trained model weights.

        :param model_path: Path to the pre-trained model weights file (.pth)
        """
        self.input_dim = 15
        self.hidden_dim = 612
        self.layer_dim = 3
        
        # Initialize the Z-location estimator model
        self.model = Zloc_Estimator(self.input_dim, self.hidden_dim, self.layer_dim)
        
        # Load the state dictionary from the file, using map_location in torch.load()
        state_dict = torch.load(CONFIG['lstm_model_path'], map_location=device)
        
        # Load the model with the state dictionary
        self.model.load_state_dict(state_dict, strict=False)        
        self.model.to(device)  # This line ensures the model is moved to the right device
        self.model.eval()  # Set the model to evaluation mode


    def predict(self, data):
        """
        Predicts the z-location based on input data.

        :param data: Input tensor of shape (batch_size, input_dim)
        :return: Predicted z-location as a tensor
        """
        with torch.no_grad():  # Disable gradient computation for deployment
            data = data.to(device)  # Move data to the appropriate device
            data = data.reshape(-1, 1, self.input_dim)  # Reshape data to (batch_size, sequence_length, input_dim)
            zloc = self.model(data)
        return zloc.cpu()  # Return the output in CPU memory for further processing

