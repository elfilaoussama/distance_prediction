import torch
import torch.nn as nn
from config import CONFIG

device = CONFIG['device']

class Zloc_Estimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)

        layersize = [306, 154, 76]
        layerlist = []
        n_in = hidden_dim
        for i in layersize:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU())
            n_in = i           
        layerlist.append(nn.Linear(layersize[-1], output_dim))

        self.fc = nn.Sequential(*layerlist)

    def forward(self, x):
        out, _ = self.rnn(x)
        output = self.fc(out[:, -1])
        return output


class LSTMModel():
    def __init__(self, path):
        self.input_dim = 9
        self.hidden_dim = 612
        self.layer_dim = 3

        self.model = Zloc_Estimator(self.input_dim, self.hidden_dim, self.layer_dim)
        self.model.load_state_dict(torch.load(path, map_location=device), strict=False)
        self.model.to(device)

    def predict(self, input_data):
        self.model.eval()
        with torch.no_grad():
            input_data = input_data.to(device)
            return self.model(input_data)
