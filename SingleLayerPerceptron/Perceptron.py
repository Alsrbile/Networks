import torch
import torchvision

import numpy
import argparse

from torch.utils.data import DataLoader

"""Machine learning System Development Flow
    
1. Build 'Data processing pipeline'
2. Build 'Model'
3. Build 'Training and Evaluation loop'
    (loss/metric, otpimizer/schedule)
4. Build 'Save and load subroutine'
"""

# Parameter
parser = argparse.ArgumentParser(description="SLP args")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--data_root", type=str, 
                    default="C:\\Users\\iles4\\Desktop\\github\\Networks\\SingleLayerPerceptron\\data")
parser.add_argument("--epoch", type=int, default=20)
args = parser.parse_args()

def train():
    return 0

def test():
    return 0

def main():
    print("<SingleLayerPerceptron_Fashion-MNIST>")
    
    # 1. Data processing pipeline (Fashion-MNIST)
    # https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
    
    train_data = torchvision.datasets.FashionMNIST(
        root = args.data_root,
        train = True,
        download = True,
        # transform = 
    )
    
    test_data = torchvision.datasets.FashionMNIST(
        root = args.data_root,
        train = False,
        download = True,
        # transform =
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)     # Do not use shuffle in test
    
    print("Train data shape:    ", train_data.data.shape)
    print("Test data shape:     ", train_data.data.shape)
    
    
    # 2. Model
   
    
    # 3. Training and Evaluation loop
    
if __name__ == "__main__":
    main()
    