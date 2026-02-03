import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return F.rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)
