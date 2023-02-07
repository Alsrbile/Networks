import argparse
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader 
from src.model import CNN

import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--device",         type=str, default="cuda:0")
parser.add_argument("--batch_size",     type=int, default=64)
parser.add_argument("--lr",         type=float, default=0.001)
args = parser.parse_args()


def main():
    print("CNN_Fashion-MNIST")
    
    # 1. Data processing pipeline (Fashion-MNIST)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])
    
    train_data = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    
    test_data = FashionMNIST(
        root="data",
        train=False,
        transform=transform,
        download=True
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    print( train_loader.dataset.data.shape)
    
    # 2. model
    model = CNN()
    model = model.to(args.device)
    
    # Training and Evaluation loop
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
if __name__ == "__main__":
    main()