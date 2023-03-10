import torch.nn as nn

class SLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 10)
        self.act = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.act(x)
        return x
    