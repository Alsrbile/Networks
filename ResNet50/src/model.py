import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels     = in_channels, 
            out_channels    = out_channels, 
            kernel_size     = kernel_size,
            stride          = stride, 
            padding         = padding
        )
        
        
        self.bn = nn.BatchNorm2d(num_features=1)
        self.gelu = nn.GELU()
        

        
    def forward(self, input):
        # conv batch act
        x = self.conv(input)
        x = self.bn(x)
        x = self.gelu(x)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        
        x = x + input
        
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        
        
    def forward(self, x):
        # 28
        # conv1
        

        # 14
        # cov2
        
        # 7
        # conv3
        
        # 3 
        # avgpool        

       
        return x


    