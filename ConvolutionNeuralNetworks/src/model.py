import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
     
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.bn = nn.BatchNorm2d(num_features=28)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
        self.fc1 = nn.Linear(64*28*28, 300)
        self.fc2 = nn.Linear(300, 10)
        self.sm = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.pool2(x)
        
        x = x.view(-1, 64*28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sm(x)
        
        return x