# from torchvision import datasets
# from torchvision import transforms

# def build_dataset(cfg, split):
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     if cfg.dataset == "MNIST":
#         dataset = datasets.MNIST(cfg.root, train = (split == "train"), download=True, transform=transform)
#     elif cfg.dataset == "CIFAR-10":
#         dataset = datasets.CIFAR10(cfg.root, train = (split == "train"), download=True, transform=transform)
#     elif cfg.dataset == "CIFAR-100":
#         dataset = datasets.CIFAR100(cfg.root, train = (split == "train"), download=True, transform=transform)
#     elif cfg.dataset == "SVHN":
#         dataset = datasets.SVHN(cfg.root, split=split, download=True, transform=transform)
#     else:
#         raise NotImplementedError
#     return dataset

from torchvision import datasets
from torchvision import transforms

def build_dataset(cfg, split):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if cfg.dataset == "MNIST":
        dataset = datasets.MNIST(cfg.root, train=(split == "train"), download=True, transform=transform)

    elif cfg.dataset == "CIFAR-10":
        dataset = datasets.CIFAR10(cfg.root, train=(split == "train"), download=True, transform=transform)

    elif cfg.dataset == "CIFAR-100":
        dataset = datasets.CIFAR100(cfg.root, train=(split == "train"), download=True, transform=transform)

    elif cfg.dataset == "SVHN":
        dataset = datasets.SVHN(cfg.root, split=split, download=True, transform=transform)

    elif cfg.dataset == "CelebA":
        celeba_split = {
            "train": "train",
            "valid": "valid",
            "test": "test"
        }.get(split)

        if celeba_split is None:
            raise ValueError("Split must be 'train', 'valid', or 'test' for CelebA")

        # Optional transforms (adjust if needed for your model)
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        dataset = datasets.CelebA(cfg.root, split=celeba_split, download=True, transform=transform)

    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} is not supported.")
    
    return dataset
