import matplotlib.pyplot as plt
import torch


# View the compressed and uncompressed images
saved_images = torch.zeros([6, 160, 160, 3])
model_saved_images = torch.zeros([6, 160, 160, 3])

def view(model, test_loader, training_steps, training_loss, device):
    get_data(model, test_loader, device)
    view_imgs(training_steps, training_loss)

def get_data(model, test_loader, device):
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

def view_imgs(training_steps, training_loss):
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
