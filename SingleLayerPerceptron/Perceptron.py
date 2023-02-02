import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional.classification import accuracy

from src.model import SLP

# Parameter
parser = argparse.ArgumentParser(description="SLP args")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--base_lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data_root", type=str, 
                    default="C:\\Users\\iles4\\Desktop\\github\\Networks\\SingleLayerPerceptron\\data")
args = parser.parse_args()


def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    # model -> train mode
    model.train()
    
    # Create average meters to measure loss and metric
    loss_mean   = MeanMetric().to(device)
    metric_mean = MeanMetric().to(device)
    
    # Train model for one epoch
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        
        # Forward
        outputs = model(inputs)
        loss    = loss_fn(outputs, targets)
        metric  = metric_fn(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        loss_mean.update(loss)
        metric_mean.update(metric)
        
        # Update learning rate
        scheduler.step()
    
    # Summarize statistics
    summary = {"loss": loss_mean.compute(), "metric": metric_mean.compute()}
    
    return summary


def test():
    return 0


"""Machine learning System Development Flow
    
1. Build 'Data processing pipeline'
2. Build 'Model'
3. Build 'Training and Evaluation loop'
    (loss/metric, otpimizer/schedule)
4. Build 'Save and load subroutine'
"""

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
    test_loader  = DataLoader(test_data, batch_size=args.batch_size)     # Do not use shuffle in test
    
    print("Train data shape:    ", train_data.data.shape)   # torch.Size([60000, 28, 28])
    print("Test data shape:     ", test_data.data.shape)    # torch.Size([10000, 28, 28])
    
    input_dim = train_data.data.shape[1]*train_data.data.shape[2]   # input_dim=784
    
    
    # 2. Model
    model = SLP(dim=input_dim)
    model.to(device=args.deivce)
    
    
    # 3. Training and Evaluation loop (optimizer, scheduler, loss function, metric function)
    optimizer   = optim.SGD(model.parameters(), lr=args.base_lr)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*len(train_loader))
    loss_fn     = nn.CrossEntropyLoss()
    metric_fn   = accuracy()
    
    
    
if __name__ == "__main__":
    main()
    