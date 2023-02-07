import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")


def main():
    print("CNN_Fashion-MNIST")
    
    # 1. Data processing pipeline (Fashion-MNIST)
    
     