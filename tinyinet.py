import os
from random import randint, shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, models
from torchvision import transforms as T
from torchvision.utils import make_grid
import torchvision.transforms as transforms


# Define device to use (CPU or GPU). CUDA = GPU support for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Define main data directory
DATA_DIR = "tiny-imagenet-200"  # Original images come in shapes of [3,64,64]

# Define training and validation data paths
IMAGE_DIR = os.path.join(DATA_DIR, "train")
TRAIN_DIR = os.path.join(DATA_DIR, "val")
VALID_DIR = os.path.join(DATA_DIR, "val")

class_names = {
    name
    for name in os.listdir(IMAGE_DIR)
    if os.path.isdir(os.path.join(IMAGE_DIR, name))
}

class_to_name_list = []
with open(os.path.join(DATA_DIR, "words.txt")) as f:
    for idx, row in enumerate(f):
        row_list = row.strip().split("\t")
        if row_list[0] in class_names:
            class_to_name_list.append([row_list[0], row_list[1]])

class_to_name_list = sorted(class_to_name_list, key=lambda element: element[0])

[x[1] for x in class_to_name_list]

# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_img_dir = os.path.join(VALID_DIR, "images")
train_img_dir = os.path.join(VALID_DIR, "images")

# Open and read val annotations text file
fp = open(os.path.join(VALID_DIR, "val_annotations.txt"), "r")
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
val_img_dict = {}
for line in data:
    words = line.split("\t")
    val_img_dict[words[0]] = words[1]
fp.close()

# Display first 10 entries of resulting val_img_dict dictionary
{k: val_img_dict[k] for k in list(val_img_dict)[:10]}

# Create subfolders (if not present) for validation images based on label,
# and move images into the respective folders
for img, folder in val_img_dict.items():
    newpath = os.path.join(val_img_dir, folder)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

# Setup function to create dataloaders for image datasets
def generate_dataloader(data, name, batch_size, transform):
    if data is None:
        return None

    # Read image files to pytorch dataset using ImageFolder, a generic data
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(
            data,
            transform=transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            ),
        )

    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(name == "train"), **kwargs
    )

    return dataloader


# Functions to display single or a batch of sample images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    imshow(make_grid(images))  # Using Torchvision.utils make_grid function


def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    random_num = randint(0, len(images) - 1)
    imshow(images[random_num])
    label = labels[random_num]
    print(
        f"Label: {label} {class_to_name_list[label.item()][1]}, Shape: {images[random_num].shape}"
    )


# # Define batch size for DataLoaders
# batch_size = 64

# # Create DataLoaders for pre-trained models (normalized based on specific requirements)
# train_loader_pretrain = generate_dataloader(TRAIN_DIR, "train", transform=None)
# val_loader_pretrain = generate_dataloader(val_img_dir, "val", transform=None)


def create_data_loader_tinyinet():
    # Create DataLoaders for pre-trained models (normalized based on specific requirements)
    train_loader_pretrain = generate_dataloader(
        TRAIN_DIR, "train", batch_size=256, transform=None
    )
    val_loader_pretrain = generate_dataloader(
        val_img_dir, "val", batch_size=256, transform=None
    )
    return train_loader_pretrain, val_loader_pretrain
