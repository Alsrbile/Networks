import torch
import argparse
import torchvision
from torch.utils.data import DataLoader
from src.model import ResNet50


from utils.yaml_args_hook import configs


def main(args):
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
    ])
    
    train_data  = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=transform, download=True)
    test_data   = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=transform, download=True)
    
    train_loader    = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader     = DataLoader(test_data, batch_size=args.batch_size)
    
    model = ResNet50()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = configs("./config/config.yaml", parser)
    
    main(args)