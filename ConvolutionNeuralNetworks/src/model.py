import torch.nn as nn

class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2) 
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        
        return x
        
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_block_11 = CNN_block(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_block_12 = CNN_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_block_13 = CNN_block(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        
        self.conv_block_21 = CNN_block(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv_block_22 = CNN_block(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv_block_23 = CNN_block(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1)
        
        self.conv_block_31 = CNN_block(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_block_32 = CNN_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_block_33 = CNN_block(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(512, 10)
        self.sm = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        
    
    def forward(self, x):

        x = self.conv_block_11(x)
        x = self.conv_block_12(x)
        x = self.conv_block_13(x)
        
        x = self.conv_block_21(x)
        x = self.conv_block_22(x)
        x = self.conv_block_23(x)
        
        x = self.conv_block_31(x)
        x = self.conv_block_32(x)
        x = self.conv_block_33(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x) 
        
        x = x.reshape(-1,512)
        x = self.fc(x)
        x = self.sm(x)

        return x