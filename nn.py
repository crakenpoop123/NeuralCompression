import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
# import numpy as np
import matplotlib.pyplot as plt
import time
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


# Init some variables about the model
learning_rate = 0.0005
num_epochs = 20
batch = 1000
saved_images = torch.randn([6, 32, 32, 3])
model_saved_images = torch.zeros([6, 32, 32, 3])

data_size = 60000
steps_per_epoch = math.floor(data_size/batch)
training_loss = [0] * steps_per_epoch * num_epochs
training_steps = list(range(steps_per_epoch * num_epochs))


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

# Variables about the model architecture
convs_out_channels = 27
convs_mid_channels = 9
convs_kernel_size = 5

input_size = 32 * 32 * 3
hidden_in_size = 4 * 4 * convs_out_channels
hidden_size = 12 * 12
large_hidden_size = 16 * 16

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Used to determine what percent of the time different parts of the code takes
        self.step_time = math.inf

        # Convolutional neural nets
        self.in_conv = nn.Conv2d(in_channels=3, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1, padding=2),
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1, padding=2),
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1, padding=2),
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1, padding=2),
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1, padding=2)
        ])

        self.shrink_convs = nn.ModuleList([
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1),
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1),
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1),
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1),
            nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_mid_channels, kernel_size=convs_kernel_size, stride=1)
        ])

        self.out_conv = nn.Conv2d(in_channels=convs_mid_channels, out_channels=convs_out_channels, kernel_size=convs_kernel_size, stride=1)
        

        # This is shown to reduce overfitting and improve conv performance
        # I have stopped using it because it causes a very blurry output
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)


        # Input hidden layers
        self.input_hiddens = nn.ModuleList([
            nn.Linear(in_features=hidden_in_size, out_features=hidden_in_size), 
            nn.Linear(in_features=hidden_in_size, out_features=hidden_in_size), 
            nn.Linear(in_features=hidden_in_size, out_features=hidden_in_size), 
            nn.Linear(in_features=hidden_in_size, out_features=hidden_in_size)
        ])

        # Linear layers
        self.input_layer = nn.Linear(in_features=hidden_in_size, out_features=hidden_size)


        self.hidden_layer = nn.Linear(in_features=hidden_in_size, out_features=hidden_size)

        self.large_hidden_layer = nn.Linear(in_features=hidden_size, out_features=large_hidden_size)

        self.large_hiddens = nn.ModuleList([
            nn.Linear(in_features=large_hidden_size, out_features=large_hidden_size), 
            nn.Linear(in_features=large_hidden_size, out_features=large_hidden_size), 
            nn.Linear(in_features=large_hidden_size, out_features=large_hidden_size), 
            nn.Linear(in_features=large_hidden_size, out_features=large_hidden_size)
        ])

        self.output_layer = nn.Linear(in_features=large_hidden_size, out_features=input_size)


    # This improves conv performance by mixing the conv block with a pool
    def conv_block(self, conv, input, residual=False):
        intermediary = conv(input)
        # intermediary = self.pool(intermediary)
        intermediary = F.relu(intermediary)

        # Make the layer a residual
        if residual:
            intermediary = intermediary + input

        return intermediary


    def linear_residual(self, linear, input):
        # Pass the output through the linear layer
        output = linear(input)

        # Add the original value to create a residual stream
        return output + input

    # Main forward pass func
    def forward(self, input):
        # print("size: ", input.size())
        intermediary = input

        # print("size: ", intermediary.size())

        start_time = time.time_ns()

        # print("in_conv: ", self.in_conv)
        intermediary = self.conv_block(self.in_conv, intermediary)

        print("In conv block took ", (time.time_ns() - start_time) / self.step_time * 100, "% of the time")

        start_time = time.time_ns()

        # Conv block
        for mid_conv in self.convs:
            # print("mid_conv: ", mid_conv)
            intermediary = self.conv_block(mid_conv, intermediary, True)

        print("mid conv block took ", (time.time_ns() - start_time) / self.step_time * 100, "% of the time")

        start_time = time.time_ns()
        
        # Conv block
        for shrink_conv in self.shrink_convs:
            # print("shrink_conv: ", shrink_conv)
            intermediary = self.conv_block(shrink_conv, intermediary)

        intermediary = self.conv_block(self.out_conv, intermediary)

        print("Shrink conv block took ", (time.time_ns() - start_time) / self.step_time * 100, "% of the time")

        # print("size: ", intermediary.size())

        # Format for linear layers
        intermediary = intermediary.view(-1, hidden_in_size)
        # print("size: ", intermediary.size())

        start_time = time.time_ns()
        
        # Large input linear block
        for hidden_in in self.input_hiddens:
            intermediary = self.linear_residual(hidden_in, intermediary)

        print("input linear took ", (time.time_ns() - start_time) / self.step_time * 100, "% of the time")


        start_time = time.time_ns()
        
        # Hidden layer
        intermediary = self.hidden_layer(intermediary)

        # print("size: ", intermediary.size())
        # Large hidden layer
        intermediary = self.large_hidden_layer(intermediary)

        print("large hidden and hidden linear took ", (time.time_ns() - start_time) / self.step_time * 100, "% of the time")

        
        start_time = time.time_ns()
        
        # Large intermediary block
        for large_hidden in self.large_hiddens:
            intermediary = self.linear_residual(large_hidden, intermediary)

        print("large intermediary block took ", (time.time_ns() - start_time) / self.step_time * 100, "% of the time")

        # Output layer
        intermediary = self.output_layer(intermediary)

        # Format to the same as inputs
        intermediary = intermediary.view(-1, 3, 32, 32)

        return intermediary



# Main training loop 
def train():

    # Zero the grad so it doesn't have any weird errors
    model.zero_grad()

    print("started training")

    for epoch in range(num_epochs):
        print("Epoch: ", epoch + 1)
        
        for i, (images, labels) in enumerate(train_loader):

            print("full step took ",model.step_time)
            
            step_start_time = time.time_ns()

            del labels
            
            # start_time = time.time_ns()

            images = images.to(device)

            # print("image loading took ", (time.time_ns() - start_time) /model.step_time * 100, "% of the time")
            
            start_time = time.time_ns()

            # Get the outputs
            output = model(images).to(device)
            
            print("model took ", (time.time_ns() - start_time) /model.step_time * 100, "% of the time")

            
            # Measure the loss
            loss = criterion(output, images)

            start_time = time.time_ns()

            # Backpropogate the error
            loss.backward()
            
            print("backprop took ", (time.time_ns() - start_time) /model.step_time * 100, "% of the time")
            
            start_time = time.time_ns()

            # Step the Adams optimizer
            optimizer.step()
            
            print("optimizer took ", (time.time_ns() - start_time) /model.step_time * 100, "% of the time")

            model.zero_grad()

            step = epoch * steps_per_epoch + i

            start_time = time.time_ns()

            print(f'loss: {loss}')
            print(f'step: {step}')

            # Add loss and step to arrays
            training_loss[step] = loss

            print("saving loss took ", (time.time_ns() - start_time) /model.step_time * 100, "% of the time")

            model.step_time = time.time_ns() - step_start_time
            print("step time: ",model.step_time)



def get_data():

    data_iter = iter(test_loader)
    data_batch, labels = next(data_iter)

    data_batch = data_batch.to(device)

    start_time = time.time_ns()

    output = model(data_batch)

    for i in range(6):
        saved_images[i] = data_batch[i].clone().detach().cpu().permute(1, 2, 0)

        model_saved_images[i] = output[i].clone().detach().cpu().permute(1, 2, 0)

    print("saving images took ", (time.time_ns() - start_time) /model.step_time * 100, "% of the time")
    


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

    plt.figure(3)
    plt.title("Training loss over time")
    plt.subplot(training_steps.numpy(), training_loss.numpy())


    plt.show()


if __name__ == '__main__':
    model = NeuralNet().to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_start_time = time.time_ns()

    train()

    print("training took: ", (time.time_ns() - train_start_time) / (10^9), " seconds")

    get_data()

    view_imgs()

    print("Done training")