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
    dataset = train_dataset, 
    batch_size=batch, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=2, 
    persistent_workers=True
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset, 
    batch_size=batch, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=2, 
    persistent_workers=True
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Init some variables
batch = 128



# Init some variables about the model architecture

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
        self.pool = nn.MaxPool2d(stride=2)

        # Upsamples the image to grow it
        self.upsample = nn.Upsample(scale_factor=2)

        # Apply relu activation function
        self.relu = nn.ReLU()

        # Apply sigmoid activation function
        self.sigmoid = nn.Sigmoid()


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
        self.out_conv = nn.Conv2d(in_channels=3, out_channels=first_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

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
            self.pool(output)
        # Add a residual stream
        elif conv.in_channels == conv.out_channels:
            output = output + input

        # Apply the ReLU activation function
        output = self.relu(output)

        return output
        


    def forward(self, input):

        # Use the first conv layer
        intermediary = self.conv_block(self.in_conv, input)

        # Use the next conv layer, the first shrink
        intermediary = self.conv_block(self.first_shrink_conv, intermediary, True)

        # Apply the encoding residual convs
        for encode in self.encode_convs:
            intermediary = self.conv_block(encode, intermediary)

        # Half the spatial dimensions
        intermediary = self.conv_block(self.second_shrink_conv, intermediary, True)

        # Shrink the channel dimensions to the choke
        intermediary = self.conv_block(self.conv_mid_choke, intermediary)

        # The data is now at the choke point

        # Grow the channel dimensions to the mid
        intermediary = self.conv_block(self.conv_choke_mid, intermediary)

        # Upsample the data
        intermediary = self.upsample(intermediary)

        # Use the accompanying grow conv
        intermediary = self.conv_block(self.first_grow_conv, intermediary)

        # Apply the decoding residual convs
        for decode in self.decode_convs:
            intermediary = self.conv_block(decode, intermediary)

        # Upsample the data
        intermediary = self.upsample(intermediary)

        # Apply the accompanying grow conv
        intermediary = self.conv_block(self.second_grow_conv, intermediary)

        # Shrink the channel dimensions to 3 (RGB)
        intermediary = self.conv_block(self.out_conv, intermediary)

        # Apply sigmoid activation function so that outputs are within 0 and 1(range for RGB values)
        output = self.sigmoid(intermediary)

        return output