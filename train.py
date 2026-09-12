import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure

import nn as Net
import view_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Variables about training

batch = 32
learning_rate = 0.001
num_epochs = 25

training_start_time = 0

# Datasets

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.Imagenette(
    root = "./data",
    size="160px",
    split="train",
    transform = transform,
    download = True
)

test_dataset = torchvision.datasets.Imagenette(
    root = "./data",
    size="160px",
    split="val",
    transform = transform,
    download = True
)

# Loaders

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=4, 
    persistent_workers=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, 
    batch_size=batch, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=4, 
    persistent_workers=True
)


# 226.63 seconds per epoch for model before doubling channels with each pool


# Variables for monitoring training

# Will increase by 1 each step
step = 0
data_size = 9469
steps_per_epoch = math.ceil(data_size / batch)

# Monitor loss over time
training_loss = []
training_steps = []



def train():
    print("Started training")

    # Zero the model gradient
    model.zero_grad()

    # Save the current time, so I can see how long training took
    training_start_time = time.time()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # I do not use labels for my model, so I can delete them to save space
            del labels

            # Move the images to the device
            images = images.to(device)

            # Get the model output
            output = model(images).to(device)

            # Measure loss
            loss = criterion(images, output)

            # Backpropogate
            loss.backward()

            # Step the optimizer
            optimizer.step()

            # Zero the gradient to prevent gradient accumulation
            model.zero_grad()

            # Update the step
            step = epoch * steps_per_epoch + i

            # Diagnostic data about the training
            print(f"Model gave a loss of: {loss.item():.4f} at step {step}")
            proportion_done = max(0.01, step / (num_epochs * steps_per_epoch))
            print(f"Training is {proportion_done * 100:.3f}% done ({(time.time() - training_start_time) * (1-proportion_done) / proportion_done:.3f} seconds left)")

            # Save the training loss
            training_loss.append(loss.item())

            # Save the training step, in fractional epochs
            training_steps.append(step / steps_per_epoch)

            # Clear unnecessary memory
            del loss
            del output
            del images



if __name__ == "__main__":
    # Import the model from nn.py
    model = Net.NeuralNet().to(device)

    # Init the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Init the loss criterion
    criterion = StructuralSimilarityIndexMeasure()
    # criterion = nn.L1Loss()

    # Save the current time, so I can see how long training took
    training_start_time = time.time()

    # Start training
    train()

    # Print the time training took
    print(f"Training took {time.time() - training_start_time:.3f} seconds")
    print(f"This is an average of {(time.time() - training_start_time)/num_epochs:.3f} seconds per epoch")

    # View stuff about the model via view_model.py
    view_model.view(model, test_loader, training_steps, training_loss, device)