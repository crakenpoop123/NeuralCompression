import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Variables about training

batch = 32

# Debugging variables
view_data_sizes = False
view_quant_sizes = False


# Variables about the model architecture

# Channel variables
first_channels = 16
mid_channels = 8
choke_channels = 6

# Quantization variables
quantized_states = 256
rand_match = 0.05

# Convolutional variables
convs_kernel_size = 5
convs_padding_size = (convs_kernel_size - 1) // 2


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Used to calculate what % of the time is being wasted on different bits of code
        self.step_time = 2^32

        # Pools the conv to shrink it
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # MSE Loss
        self.mse = nn.MSELoss()
        
        # Used for dynamically quantizing the compressed hidden state of the model
        self.quantized_vals = torch.randn(quantized_states, choke_channels).to(device)

        self.quantized_restart_active = True


        # Upsamples the image to grow it
        self.upsample = nn.Upsample(scale_factor=2)

        # Apply relu activation function
        self.relu = nn.LeakyReLU(0.1)


        # Half the spatial dimensions
        self.shrink_convs = nn.ModuleList([
             nn.Conv2d(in_channels=3, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size), 
             nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels * 2, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size), 
             nn.Conv2d(in_channels=mid_channels * 2, out_channels=mid_channels * 4, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size), 
            #  nn.Conv2d(in_channels=mid_channels * 4, out_channels=mid_channels * 8, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)
        ])

        # Double the spatial dimensions
        self.grow_convs = nn.ModuleList([
            #  nn.Conv2d(in_channels=mid_channels * 8, out_channels=mid_channels * 4, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size), 
             nn.Conv2d(in_channels=mid_channels * 4, out_channels=mid_channels * 2, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size), 
             nn.Conv2d(in_channels=mid_channels * 2, out_channels=mid_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size), 
             nn.Conv2d(in_channels=mid_channels, out_channels=3, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)
        ])


        # Changes the channel dimension from mid to choke
        self.conv_mid_choke = nn.Conv2d(in_channels=mid_channels * 4, out_channels=choke_channels, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

        # Changes the channel dimension from choke to mid
        self.conv_choke_mid = nn.Conv2d(in_channels=choke_channels, out_channels=mid_channels * 4, kernel_size=convs_kernel_size, stride=1, padding=convs_padding_size)

    def get_most_similar_state(self, input):
        # Normalise quantized values and input
        norm_quantized = F.normalize(self.quantized_vals, p=2, dim=1)
        norm_input = F.normalize(input, p=2, dim=1)

        if view_quant_sizes: 
            print("quantized_vals size: ", norm_quantized.size())
            print("intermediary size: ", norm_input.size())

        distances = torch.cdist(
            norm_quantized.unsqueeze(0), 
            norm_input.unsqueeze(0), 
            p=2
        ).squeeze(0)

        # Find the state with the lowest loss
        best_state = torch.argmin(distances, dim=0).to(device)
        

        return best_state

    def update_quantized_states(self, input, best_matches, lr=0.01):
        # Counts up all the values
        counts = torch.bincount(best_matches, minlength=self.quantized_vals.size(0)).float().unsqueeze(1)

        print("quantized_restart_active: ", True if self.quantized_restart_active else False)

        # Check if the quantized restart stuff is active
        if self.quantized_restart_active:
            # Find the indices of all quantized vals that were not picked
            zero_indices = (counts.squeeze() == 0).nonzero(as_tuple=True)[0]

            # Check that zero_indices is non_empty
            if len(zero_indices) > 0:
                # Get a random vector from the input vals
                random_idx = torch.randint(0, input.size(0), (len(zero_indices),), device='cuda')

                # Update the value of a random quantized val to this random vector
                norm_injected_vals = F.normalize(input[random_idx], p=2, dim=1)
                self.quantized_vals[zero_indices] = norm_injected_vals
        
        # Used to sum all vectors
        total_assigned_vectors = torch.zeros_like(self.quantized_vals)
        
        # Sums the vectors along dimension 0, 
        # using best_matches a mask so that only the values that were actually the best get added
        total_assigned_vectors.index_add_(0, best_matches.long(), input)
        
        # Preven division by 0
        mask = counts > 0
        # Averages the vectors
        average_vectors = torch.where(mask, total_assigned_vectors / counts, self.quantized_vals)
        
        # Use an Exponential Moving Average to shift the quantized values
        updated_vals = self.quantized_vals * (1 - lr) + average_vectors * lr

        self.quantized_vals.copy_(F.normalize(updated_vals, p=2, dim=1))

    
    def quantize(self, flattened_input):
        # if view_quant_sizes: 
        #     print("Got input as size: ", input.size())

        # flattened_input = input.reshape(-1, 6).to(device)

        if view_quant_sizes: 
            print("Got flattened input for quantization to: ", flattened_input.size())

        # Get the best quantization match
        best_matches = self.get_most_similar_state(flattened_input).to(device, dtype=torch.uint8)


        # Note: best_matches is what the very smallest choke point for the data is

        if view_quant_sizes: 
            print("Got best matches as size: ", best_matches.size())

        print(f"Best matches num unique items over total items: {len(torch.unique(best_matches))}/{len(best_matches)}")

    
        if len(torch.unique(best_matches)) == 256:
            self.quantized_restart_active = False


        # Get the quantized versions of the vectors
        quantized_vectors = self.quantized_vals[best_matches.long()]

        intermediary = flattened_input + (quantized_vectors - flattened_input).detach()

        if view_quant_sizes: 
            print("Intermediary became size: ", best_matches.size())


        quantized_lr = 0.05 

        # Shift the quantized vals slightly in the direction of the input
        self.update_quantized_states(intermediary.detach(), best_matches, quantized_lr)


        if view_quant_sizes: 
            print("Viewed intermediary as: ", intermediary.size())

        return intermediary



    def conv_block(self, conv, input, sample=0):
        # Run the conv
        output = conv(input)

        # Pool the matrix
        if sample == 1:
            output = self.pool(output)
        # Upsample the matrix
        elif sample == -1:
            output = self.upsample(output)

        # Add a residual stream
        elif conv.in_channels == conv.out_channels:
            output = output + input

        # Apply the ReLU activation function
        output = self.relu(output)

        return output
        
    def encode(self, input):
        if view_data_sizes: 
            print(f"{batch} * 3 * 160 * 160: ", input.size())


        intermediary = input

        # Half the spatial dimensions
        for shrink_conv in self.shrink_convs:
            intermediary = self.conv_block(shrink_conv, intermediary, 1)
            
            if view_data_sizes: 
                print("Too lazy to write desired size but size is: ", intermediary.size())

        if view_data_sizes: 
            print(f"{batch} * {mid_channels * 8} * 10 * 10: ", intermediary.size())
        

        # Shrink the channel dimensions to the choke
        intermediary = self.conv_block(self.conv_mid_choke, intermediary)
        
        if view_data_sizes: 
            print(f"{batch} * {choke_channels} * 10 * 10: ", intermediary.size())

        return intermediary

    def decode(self, input):
        if view_data_sizes: 
            print(f"{batch} * {choke_channels} * 10 * 10: ", input.size())

        # Grow the channel dimensions to the mid
        intermediary = self.conv_block(self.conv_choke_mid, input)

        if view_data_sizes: 
            print(f"{batch} * {mid_channels * 8} * 10 * 10: ", intermediary.size())

        # Iterate over grow convs
        for grow_conv in self.grow_convs:
            # Upsample the data
            intermediary = self.upsample(intermediary)

            # Apply the accompanying grow conv
            intermediary = self.conv_block(grow_conv, intermediary)

            if view_data_sizes: 
                print("Too lazy to write desired size but size is: ", intermediary.size())


        if view_data_sizes: 
            print(f"{batch} * 3 * 160 * 160: ", intermediary.size())

        # Apply sigmoid activation function so that outputs are within 0 and 1(range for RGB values)
        output = torch.sigmoid(intermediary)

        return output


    def forward(self, input):
        if view_data_sizes: 
            print("--------------------------------------")

        # Encode the image
        intermediary = self.encode(input)

        # Print the middle data
        if view_data_sizes: 
            print(f"The data is now at the choke point")

        # intermediary = self.quantize(intermediary.permute(0, 2, 3, 1).reshape(-1, 6))

        # intermediary = intermediary.view(-1, 10, 10, 6).permute(0, 3, 1, 2)

        # Decode the image
        output = self.decode(intermediary)

        if view_data_sizes: 
            print("--------------------------------------")

        return output