import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.softmax(self.fc3(x))
                
        return x
    