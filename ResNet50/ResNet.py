import torch
import argparse
import torchvision

def main():
    
    transform = torchvision.transforms.Compose(
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    )
    
    train_data  = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=transform)
    test_data   = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=transform)
    
    train_loader = 
    
    
if __name__ == "__main__":
    main()