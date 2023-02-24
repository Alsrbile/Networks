import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels     = in_channels, 
            out_channels    = out_channels, 
            kernel_size     = kernel_size,
            stride          = stride, 
            padding         = 1
        )
        
        self.conv2 = nn.Conv2d(
            in_channels     = out_channels, 
            out_channels    = out_channels, 
            kernel_size     = kernel_size,
            stride          = 1, 
            padding         = 1
        )
        
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.gelu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.2) 
        
        self.downsample = downsample
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
        
    def forward(self, input):
        # conv batch act
        
        x = self.conv1(input)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.drop(x)
        
        x = self.conv2(x)
        x = self.bn(x)
        if self.downsample is not None:
            input = self.downsample(input)
        
        
        x = x + input
        
        x = self.gelu(x)
        x = self.drop(x)
        
        return x
    
    



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.in_ch = 1
        
        # self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._layer(in_ch=1, out_ch=32, num=2, stride=2)
        self.layer2 = self._layer(in_ch=32, out_ch=64, num=3, stride=2)
        self.layer3 = self._layer(in_ch=64, out_ch=128, num=3, stride=2)
        self.layer4 = self._layer(in_ch=128, out_ch=256, num=3, stride=2)
        
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)
        
        self.conv1 = nn.Conv2d(1, 32,7,2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.2) 
        
    def _layer(self, in_ch, out_ch, num, stride):
        downsample = None
        layers = []
        
        if stride == 2: 
            downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch, out_channels= out_ch, kernel_size=1, stride=2),
                    nn.BatchNorm2d(out_ch)
            )
            
        layers.append(
            BasicBlock(in_ch, out_ch, kernel_size=3, stride=2, padding=1, downsample=downsample)
        )
        
        for _ in range(1, num):
            layers.append(
                BasicBlock(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
            )
        
        return nn.Sequential(*layers)
        
        
        
        
    def forward(self, x):
        
        
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.maxpool(x)
        
        # layer1
        # x = self.layer1(x)
        # B, 32, 14, 14
        
        # 14
        # layer2
        x = self.layer2(x)
        
        # 7
        # layer3
        x = self.layer3(x)
        
        x = self.layer4(x)
        # 3 
        # avgpool        
        x = self.avgpool(x)
  
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.softmax(x)
       
        return x


    