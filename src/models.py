import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=4):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        rnn_output, hidden_state = self.rnn(x)
        last_hidden_state = rnn_output[:, -1, :]
        return self.output_layer(last_hidden_state)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_output, hidden_state = self.lstm(x)
        last_hidden_state = lstm_output[:, -1, :]
        return self.output_layer(last_hidden_state)