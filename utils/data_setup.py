import torchvision
from torch.utils.data import Subset,DataLoader
import torch
from torch.utils.data import Subset
from pathlib import Path

def create_dataloaders(
    train_dir : Path,
    train_transformer : torchvision.transforms.Compose,
    test_transformer : torchvision.transforms.Compose,
    test_dir : Path = None,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    train_and_test_seperate : bool = True,

    ):


    if not train_and_test_seperate:

        train_data_full = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transformer)
        test_data_full = torchvision.datasets.ImageFolder(root=train_dir, transform=test_transformer)

        n_dataset = len(train_data_full)
        test_ratio = 0.2
        n_test = int(n_dataset * test_ratio)
        n_train = n_dataset - n_test

        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(generator=g, n=n_dataset).tolist()

        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train_dataset = Subset(indices=train_idx, dataset=train_data_full)
        test_dataset = Subset(indices=test_idx, dataset=test_data_full)
        class_names = train_data_full.classes

    else:
        if test_dir is None:
            raise ValueError("Eğer train_and_test_seperate=True ise, 'test_dir' parametresini sağlamalısınız!")

        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transformer)
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transformer)
        class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


    return train_loader, test_loader,class_names
