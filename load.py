import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torchvision
import torchvision.transforms as transforms
import math

import nn as Net
import view_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Variables about training

batch = 32


# Datasets

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])


test_dataset = torchvision.datasets.Imagenette(
    root = "./data",
    size="160px",
    split="val",
    transform = transform,
    download = True
)

# Loaders

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, 
    batch_size=batch, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=4, 
    persistent_workers=True
)


# Variables for monitoring training

# Will increase by 1 each step
step = 0
data_size = 9469
steps_per_epoch = math.ceil(data_size / batch)



if __name__ == "__main__":
    # Load the model
    PATH = "./models/model.pth"

    loaded_model = torch.load(PATH, weights_only=False).to(device)

    loaded_model.eval()

    # Import the model from nn.py
    model = Net.NeuralNet().to(device)


    # View stuff about the model via view_model.py
    view_model.view(model, test_loader, [], [], device)