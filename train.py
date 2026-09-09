import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import math


# Variables about training

batch = 128
learning_rate = 0.001
num_epochs = 1


# Datasets

train_dataset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

test_dataset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = False,
    transform = transforms.ToTensor(),
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


# Variables for monitoring training

# Will increase by 1 each step
step = 0
data_size = 50000
steps_per_epoch = math.ceil(data_size / batch)

# View the compressed and uncompressed images
saved_images = torch.zeros([6, 32, 32, 3])
model_saved_images = torch.zeros([6, 32, 32, 3])
# Monitor loss over time
training_loss = []
training_steps = []

# Debugging variables
view_data_sizes = False


# Variables about the model architecture

first_channels = 16
mid_channels = 9
choke_channels = 8

convs_kernel_size = 5
convs_padding_size = (convs_kernel_size - 1) // 2


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Used to calculate what % of the time is being wasted on different bits of code
        self.step_time = math.inf

        # Pools the conv to shrink it
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsamples the image to grow it
        self.upsample = nn.Upsample(scale_factor=2)

        # Apply relu activation function
        self.relu = nn.LeakyReLU(0.1)

        # Apply sigmoid activation function
        # Replaced with torch.sigmoid
        # self.sigmoid = nn.Sigmoid()


        # Half the spatial dimensions
        self.first_shrink_conv = nn.Conv2d(in_channels=first_channels, out_channels=first_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

        # Half the spatial dimensions, again
        self.second_shrink_conv = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

        # Accompanies the first upsample
        self.first_grow_conv = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

        # Accompanies the second upsample
        self.second_grow_conv = nn.Conv2d(in_channels=first_channels, out_channels=first_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)


        # Increase the channel dimensions from 3 (RGB) to first
        self.in_conv = nn.Conv2d(in_channels=3, out_channels=first_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

        # Decrease the channel dimensions from first to 3 (RGB)
        self.out_conv = nn.Conv2d(in_channels=first_channels, out_channels=3, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

        # Changes the channel dimension from mid to choke
        self.conv_mid_choke = nn.Conv2d(in_channels=mid_channels, out_channels=choke_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

        # Changes the channel dimension from choke to mid
        self.conv_choke_mid = nn.Conv2d(in_channels=choke_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)


        # Apply a residual stream for encoding
        self.encode_convs = nn.ModuleList([
            nn.Conv2d(in_channels=first_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)
        ])

        # Apply a residual stream for decoding
        self.decode_convs = nn.ModuleList([
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size),
            nn.Conv2d(in_channels=mid_channels, out_channels=first_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)
        ])

    def conv_block(self, conv, input, pool=False):
        # Run the conv
        output = conv(input)

        # Pool the matrix
        if pool:
            output = self.pool(output)
        # Add a residual stream
        elif conv.in_channels == conv.out_channels:
            output = output + input

        # Apply the ReLU activation function
        output = self.relu(output)

        return output
        


    def forward(self, input):
        if view_data_sizes: 
            print(f"{batch} * 3 * 32 * 32: ", input.size())

        # Use the first conv layer
        intermediary = self.conv_block(self.in_conv, input)

        if view_data_sizes: 
            print(f"{batch} * {first_channels} * 32 * 32: ", intermediary.size())

        # Use the next conv layer, the first shrink
        intermediary = self.conv_block(self.first_shrink_conv, intermediary, True)

        if view_data_sizes: 
            print(f"{batch} * {first_channels} * 16 * 16: ", intermediary.size())

        # Apply the encoding residual convs
        for encode in self.encode_convs:
            intermediary = self.conv_block(encode, intermediary)

        if view_data_sizes: 
            print(f"{batch} * {mid_channels} * 16 * 16: ", intermediary.size())

        # Half the spatial dimensions
        intermediary = self.conv_block(self.second_shrink_conv, intermediary, True)

        if view_data_sizes: 
            print(f"{batch} * {mid_channels} * 8 * 8: ", intermediary.size())

        # Shrink the channel dimensions to the choke
        intermediary = self.conv_block(self.conv_mid_choke, intermediary)

        if view_data_sizes: 
            print(f"{batch} * {choke_channels} * 8 * 8: ", intermediary.size())

        # The data is now at the choke point

        # Grow the channel dimensions to the mid
        intermediary = self.conv_block(self.conv_choke_mid, intermediary)

        if view_data_sizes: 
            print(f"{batch} * {mid_channels} * 8 * 8: ", intermediary.size())

        # Upsample the data
        intermediary = self.upsample(intermediary)

        # Use the accompanying grow conv
        intermediary = self.conv_block(self.first_grow_conv, intermediary)

        if view_data_sizes: 
            print(f"{batch} * {mid_channels} * 16 * 16: ", intermediary.size())

        # Apply the decoding residual convs
        for decode in self.decode_convs:
            intermediary = self.conv_block(decode, intermediary)

        if view_data_sizes: 
            print(f"{batch} * {first_channels} * 16 * 16: ", intermediary.size())

        # Upsample the data
        intermediary = self.upsample(intermediary)

        # Apply the accompanying grow conv
        intermediary = self.conv_block(self.second_grow_conv, intermediary)

        if view_data_sizes: 
            print(f"{batch} * {first_channels} * 32 * 32: ", intermediary.size())

        # Shrink the channel dimensions to 3 (RGB)
        intermediary = self.conv_block(self.out_conv, intermediary)

        if view_data_sizes: 
            print(f"{batch} * 3 * 32 * 32: ", intermediary.size())

        # Apply sigmoid activation function so that outputs are within 0 and 1(range for RGB values)
        output = torch.sigmoid(intermediary)

        return output

def train():
    print("Started training")

    # Zero the model gradient
    model.zero_grad()

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


            # Diagnostic data about the training
            print(f"Model gave a loss of: {loss.item():.4f} at step {step}")

            # Save the training loss
            training_loss.append(loss.item())

            # Save the training step, in fractional epochs
            training_steps.append(step / steps_per_epoch)

            # Increase the step count
            step += 1


def get_data():
    # Iterate the test data
    data_iter = iter(test_loader)
    # get a single batch
    data_batch, labels = next(data_iter)

    data_batch = data_batch.to(device)

    # Run the batch through the model
    output = model(data_batch)

    # Iterate through 6 images of each type
    for i in range(6):
        # Update the normal images
        saved_images[i] = data_batch[i].clone().detach().cpu().permute(1, 2, 0)

        # Update the compressed and uncompressed images
        model_saved_images[i] = output[i].clone().detach().cpu().permute(1, 2, 0)

def view_imgs():
    # Show the uncompressed images
        plt.figure(1)
        plt.title("Original (never compressed) images")
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(saved_images[i])
    
    
        # Show the compressed then uncompressed images
        plt.figure(2)
        plt.title("Modified (compressed then uncompressed) images")
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(model_saved_images[i])


        # Show the training loss
        plt.figure(3)
        plt.title("Training loss over time")
        plt.plot(training_steps, training_loss)
        plt.xlabel("Epoch number")
        plt.ylabel("Loss")
    
    
        plt.show()

if __name__ == "__main__":
    # Init the model
    model = NeuralNet().to(device)

    # Init the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Init the loss criterion
    criterion = nn.L1Loss()

    # Save the current time, so I can see how long training took
    training_start_time = time.time()

    # Start training
    train()

    # Print the time training took
    print("Training took ", time.time() - training_start_time)

    print()

    get_data()
    
    view_imgs()