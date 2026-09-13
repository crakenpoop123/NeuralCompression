import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torchvision
import torchvision.transforms as transforms
import time
import math
import torch.nn as nn
# from torchmetrics.image import StructuralSimilarityIndexMeasure

import nn as Net
import view_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Variables about training

batch = 32
learning_rate = 0.001
learning_rate_reduction = 0.1
max_lr_reductions = 2
num_epochs = 25

training_start_time = 0
#  If the loss hasn't decreased by this * loss after 1 epoch, learning rate gets decreased
average_loss_margin = 0.01

view_training_progress = False

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


# Variables for monitoring training

# Used to track loss and some other stuff
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
            loss = criterion(output, images)

            # Backpropogate
            loss.backward()

            # Step the optimizer
            optimizer.step()

            # Zero the gradient to prevent gradient accumulation
            model.zero_grad()

            # Update the step
            step = epoch * steps_per_epoch + i

            # Diagnostic data about the training
            if view_training_progress:
                print(f"Model gave a loss of: {loss.item():.4f} at step {step}")

                proportion_done = max(0.01, step / (num_epochs * steps_per_epoch))

                print(f"Training is {proportion_done * 100:.3f}% done ({(time.time() - training_start_time) * (1-proportion_done) / proportion_done:.3f} seconds left)")

            # Save the training loss
            training_loss.append(loss.item())

            # Save the training step, in fractional epochs
            training_steps.append(step / steps_per_epoch)


        # Update the step
        step = (epoch + 1) * steps_per_epoch

        avg_loss = average_loss(step - 1, steps_per_epoch)

        # Diagnostic data about the training
        print(f"Model gave a loss of: {avg_loss} at step {step}")

        proportion_done = max(0.01, step / (num_epochs * steps_per_epoch))

        print(f"Training is {proportion_done * 100:.3f}% done ({(time.time() - training_start_time) * (1-proportion_done) / proportion_done:.3f} seconds left)")

        
        # Decrease lr if loss isn't improving, once per epoch
        check_for_stability(step - 1)
        
    
    # Clear unnecessary memory when training ends
    del loss
    del output
    del images


# Check if the model has stabilised
def check_for_stability(step, num_epochs = 1):
    last_epoch_loss = average_loss(step - num_epochs * steps_per_epoch, steps_per_epoch)

    curr_epoch_loss = average_loss(step, steps_per_epoch)

    # Check there are enough loss values to avg over
    if last_epoch_loss == "False" or curr_epoch_loss == "False":
        return False

    print(f"loss changed by {curr_epoch_loss / last_epoch_loss - 1} of prev")

    # Check if loss has stopped improving
    if curr_epoch_loss > (1-average_loss_margin) * last_epoch_loss:

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            # Ensure the lr doesn't drop too low
            if param_group["lr"] > 10^-(max_lr_reductions) * learning_rate:
                param_group["lr"] *= learning_rate_reduction
                print("Learning rate changed to: ", param_group["lr"])



def average_loss(step, num_to_avg):
    # Prevent negative indexing
    if step - num_to_avg + 1 < 0:
        return "False"

    # Prevent div by 0
    num_to_avg = max(1, num_to_avg)

    total = 0

    # Sum all training losses
    for i in range(num_to_avg):
        total += training_loss[step - i]

    # Average using mean
    mean = total / num_to_avg

    return mean


if __name__ == "__main__":
    # Import the model from nn.py
    model = Net.NeuralNet().to(device)

    # Init the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Init the loss criterion
    # criterion = StructuralSimilarityIndexMeasure().to(device)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=0.5)

    # Save the current time, so I can see how long training took
    training_start_time = time.time()

    # Start training
    train()

    
    # Print the time training took
    print(f"Training took {time.time() - training_start_time:.3f} seconds")
    print(f"This is an average of {(time.time() - training_start_time)/num_epochs:.3f} seconds per epoch")


    # Save the model
    PATH = "./models/model2.pth"
    torch.save(model, PATH)

    # View stuff about the model via view_model.py
    view_model.view(model, test_loader, training_steps, training_loss, device)
