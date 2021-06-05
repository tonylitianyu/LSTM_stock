import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim). requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim). requires_grad_()
        out, (hn, cn) = self.lstm(x,(h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out






