### ----------------------------------------------------
### Various Model Architectures for future usage (torch) 
### ----------------------------------------------------

from torch import nn

class MLP(nn.Module):
    def __init__(self, n_features, hidden_size=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, X):
        return self.layers(X).squeeze(-1)