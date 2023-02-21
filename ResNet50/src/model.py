import torch.nn as nn

def get_resnet(name):
    
    model = {"resnet18":ResNet18,
             "resnet50":ResNet50}
    
    if name not in model.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    
    return model[name]

class CNN_block(nn.Module):
    def __init__(self):
        super(self, ResNet18).__init__()

        self.conv = nn.Conv2d(in_channels=784)
        
    def forward(self, x):
        
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super(self, ResNet18).__init__()
        
        #self.conv1 = nn.Conv2d(in_channels=784, out_channels=)
        
    def forward(self, x):
        
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(self, ResNet50).__init__()
        
        
    def forward(self, x):
        
        # 
                
        return x    
    