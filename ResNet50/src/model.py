import torch.nn as nn



class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNN_block, self).__init__()

        
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        
        
    def forward(self, x):
        
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv = CNN_block(1, 64, 3, 2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool(x)
        print(x.shape)
        
        return x


    