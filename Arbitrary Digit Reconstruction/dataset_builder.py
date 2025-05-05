import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

def build_dataset(cfg, split):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if cfg.dataset == "MNIST":
        dataset = datasets.MNIST(cfg.root, train = (split == "train"), download=True, transform=transform)
    elif cfg.dataset == "CIFAR-10":
        dataset = datasets.CIFAR10(cfg.root, train = (split == "train"), download=True, transform=transform)
    elif cfg.dataset == "CIFAR-100":
        dataset = datasets.CIFAR100(cfg.root, train = (split == "train"), download=True, transform=transform)
    elif cfg.dataset == "SVHN":
        dataset = datasets.SVHN(cfg.root, split=split, download=True, transform=transform)
    else:
        raise NotImplementedError
    return dataset

def build_dataset_pruned(cfg):
    transform = transforms.Compose([
        transforms.Pad(1),
        transforms.ToTensor()
    ])
    
    # Load the full MNIST test set
    dataset = datasets.MNIST(cfg.root, train=False, download=True, transform=transform)
    
    # Randomly select 1000 indices
    all_indices = torch.randperm(len(dataset))[:1000]  # Random permutation, take first 1000
    pruned_dataset = Subset(dataset, all_indices)
    
    return pruned_dataset    

def build_dataset_specific(cfg, number):
    transform = transforms.Compose([
        transforms.Pad(1),
        transforms.ToTensor()
    ])
    
    # Load the full training MNIST dataset
    dataset = datasets.MNIST(cfg.root, train=True, download=True, transform=transform)

    # Get indices of all samples where the target == number
    indices = [i for i, (img, label) in enumerate(dataset) if label == number]

    # Return a subset of the dataset with only those indices
    subset = Subset(dataset, indices)
    
    return subset
