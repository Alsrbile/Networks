import torch
import os
import argparse
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader 
from src.model import CNN
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.classification import accuracy
from torchmetrics.aggregation import MeanMetric
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--device",         type=str,   default="cuda:0")
parser.add_argument("--batch_size",     type=int,   default=64)
parser.add_argument("--lr",             type=float, default=0.001)
parser.add_argument("--epochs",         type=int,   default=50)
parser.add_argument("--log_dir",        type=str,   default="log")
parser.add_argument("--checkpoint_dir",     type=str, default="checkpoint")
args = parser.parse_args()

def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    
    loss_mean   = MeanMetric().to(device)
    metric_mean = MeanMetric().to(device)
    
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss    = loss_fn(outputs, targets)
        metric  = metric_fn(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss)
        metric_mean.update(metric)
        
        scheduler.step()
    
    summary = {"loss": loss_mean.compute(), "metric": metric_mean.compute()}
    
    return summary

def evaluate(loader, model, loss_fn, metric_fn, device):
    # model -> evaluation mode
    model.eval()
    
    # Create average meters to measure loss and accuracy
    loss_mean   = MeanMetric().to(device)
    metric_mean = MeanMetric().to(device)
    
    # Evaluate mode for one epoch
    for inputs, targets, in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        
        # Forward
        with torch.no_grad():
            outputs = model(inputs)
        loss    = loss_fn(outputs, targets)
        metric  = metric_fn(outputs, targets)
        
        # Update statistics
        loss_mean.update(loss)
        metric_mean.update(metric)
    
    # Summarize statistics
    summary = {"loss": loss_mean.compute(), "metric": metric_mean.compute()}
    
    return summary

def main():
    name ="CNN_Fashion-MNIST" 
    print(name)
    os.makedirs(f"{args.checkpoint_dir}", exist_ok=True)
    
    # 1. Data processing pipeline (Fashion-MNIST)
    
    transform = T.Compose([
        T.ToTensor(),       # PIL image -> Tensor
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
    
    print(train_loader.dataset.data.shape)
    
    # 2. model
    model = CNN()
    model = model.to(args.device)
    
    # 3. Training and Evaluation loop
    optimizer   = optim.Adam(model.parameters(), lr=args.lr)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*len(train_loader))
    loss_fn     = nn.CrossEntropyLoss()
    metric_fn   = accuracy
    
    train_logger    = SummaryWriter(f"{args.log_dir}/train")
    test_logger     = SummaryWriter(f"{args.log_dir}/test")
    
    if os.path.isfile(f"{args.checkpoint_dir}"):
        state_dict  = torch.load(f"{args.checkpoint_dir}/{name}_last.pth")
        start_epoch = state_dict["epoch"]
        model.load_state_dict(state_dict=["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, args.epochs):
        train_summary   = train(train_loader, model, optimizer, scheduler,
                                loss_fn, metric_fn, args.device)

        test_summary    = evaluate(test_loader, model, 
                                loss_fn, metric_fn, args.device)
        
        train_logger.add_scalar("Loss", train_summary["loss"], epoch+1)
        train_logger.add_scalar("Acc", train_summary["metric"], epoch+1)
        test_logger.add_scalar("Loss", test_summary['loss'], epoch+1)
        test_logger.add_scalar("Acc", test_summary['metric'], epoch+1)
        
        print(f"Epoch {epoch+1}: "
              + f"Train loss {train_summary['loss']:.04f}, "
              + f"Train acc {train_summary['metric']:.04f}, "
              + f"Test loss {test_summary['loss']:.04f}, "
              + f"Test acc {test_summary['metric']:.04f}")
        
        # 4. Save and load subroutine
        state_dict = {  
            "epoch":        epoch+1,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
        }
        
        torch.save(state_dict, f"{args.checkpoint_dir}/{name}_last.pth")
    
    train_logger.close()
    test_logger.close()
    
if __name__ == "__main__":
    main()