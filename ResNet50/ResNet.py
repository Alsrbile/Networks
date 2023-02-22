import torch
import os
import argparse
import torchvision
from torch.utils.data import DataLoader
from src.model import ResNet
import torch.optim as optim
import torch.nn as nn
from torchmetrics.functional.classification import accuracy
from torch.utils.tensorboard import SummaryWriter
from utils.yaml_args_hook import configs
from torchmetrics.aggregation import MeanMetric

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
    model.eval()
    
    loss_mean   = MeanMetric().to(device)
    metric_mean = MeanMetric().to(device)
    
    for inputs, targets, in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        loss    = loss_fn(outputs, targets)
        metric  = metric_fn(outputs, targets)
        
        loss_mean.update(loss)
        metric_mean.update(metric)
    
    summary = {"loss": loss_mean.compute(), "metric": metric_mean.compute()}
    
    return summary


def main(args):
    name ="ResNet_Fashion-MNIST" 
    print(name)
    os.makedirs(f"{args.checkpoint_dir}", exist_ok=True)
    
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
    
    model = ResNet()
    model = model.to(args.device)
    
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
    parser = argparse.ArgumentParser()
    args = configs("./config/config.yaml", parser)
    
    main(args)