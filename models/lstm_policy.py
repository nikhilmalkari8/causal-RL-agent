import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(LSTMPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx=None):
        # x: (batch, seq_len=1, input_dim)
        if hx is None:
            hx = (torch.zeros(1, x.size(0), self.hidden_dim).to(x.device),
                  torch.zeros(1, x.size(0), self.hidden_dim).to(x.device))
        lstm_out, hx = self.lstm(x, hx)  # lstm_out shape: (batch, seq_len, hidden_dim)
        lstm_out = lstm_out[:, -1, :]    # Take last time step output

        action_logits = self.fc_actor(lstm_out)
        state_value = self.fc_critic(lstm_out)

        return action_logits, state_value, hx
