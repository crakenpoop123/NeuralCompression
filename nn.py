import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Init some variables about the model
learning_rate = 0.002
num_epochs = 15
batch = 1000
saved_images = torch.zeros([6, 32, 32, 3])
model_saved_images = torch.zeros([6, 32, 32, 3])

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
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset, 
    batch_size=batch, 
    shuffle=True
)

# Variables about the model architecture
convs_out_channels = 27

input_size = 32 * 32 * 3
hidden_in_size = 26 * 26 * convs_out_channels
hidden_size = 16 * 16
large_hidden_size = 16 * 16

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Convolutional neural nets
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1), 
            nn.Conv2d(in_channels=9, out_channels=convs_out_channels, kernel_size=3, stride=1)
        ])

        # This is shown to reduce overfitting and improve conv performance
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        # Linear layers
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size)

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
    def conv_block(self, input, index):
        intermediary = self.convs[index](input)
        intermediary = self.pool(intermediary)

        return intermediary


    # Main forward pass func
    def forward(self, input):
        # print("size: ", input.size())
        intermediary = input

        # print("size: ", intermediary.size())

        # Conv block
        for i in range(len(self.convs)):
            intermediary = self.conv_block(intermediary, i)

        # print("size: ", intermediary.size())

        # Format for linear layers
        intermediary = intermediary.view(-1, hidden_in_size)
        # print("size: ", intermediary.size())
        # Hidden layer
        intermediary = self.hidden_layer(intermediary)

        # print("size: ", intermediary.size())
        # Large hidden layer
        intermediary = self.large_hidden_layer(intermediary)

        
        # Large intermediary block
        for large_hidden in self.large_hiddens:
            intermediary = large_hidden(intermediary)

        # print("size: ", intermediary.size())
        # Output layer
        intermediary = self.output_layer(intermediary)

        # print("size: ", intermediary.size())
        # Format to the same as inputs
        intermediary = intermediary.view(-1, 3, 32, 32)

        # print("output size(): ", intermediary.size())

        return intermediary



# Main training loop 
def train():
    # Zero the grad so it doesn't have any weird errors
    model.zero_grad()

    print("started training")

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)

        # Adjust the learning rate so that it does large modifications at first, before slowly finetuning to the minimum
        # for param in optimizer.param_groups:
        #     param = learning_rate / (epoch + 1)
        
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)

            # Get the outputs
            output = model(images).to(device)
            
            # Measure the loss
            loss = criterion(output, images) + criterion_two(output, images)
            # loss = criterion(output, images)
            # loss = criterion_two(output, images)

            # Backpropogate the error
            loss.backward()

            # Step the Adams optimizer
            optimizer.step()

            model.zero_grad()

            print(f'loss: {loss}')


def view_imgs():

    # Show the uncompressed images
    plt.figure(1)
    plt.title("original (never compressed) images")
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(saved_images[i])


    # Show the compressed then uncompressed images
    plt.figure(2)
    plt.title("modified (compressed then uncompressed) images")
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(model_saved_images[i])


    plt.show()


if __name__ == '__main__':
    model = NeuralNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion_two = nn.MSELoss()

    train()

    view_imgs()

    print("Done training")