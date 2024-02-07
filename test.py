# import numpy as np

# def CustomConv2d(in_channels, out_channels, kernel_size, stride, padding):
#     def convolve(input_map, kernel):
#         # Get input map dimensions
#         in_height, in_width, in_channels = input_map.shape

#         # Get kernel dimensions
#         kernel_height, kernel_width, input_channels = kernel.shape

#         # Calculate output dimensions
#         out_height = (in_height - kernel_height + 2 * padding) // stride + 1
#         out_width = (in_width - kernel_width + 2 * padding) // stride + 1

#         # Initialize the output feature map with multiple channels
#         output_map = np.zeros((out_height, out_width, out_channels))

#         # Pad the input map
#         input_map_padded = np.pad(input_map, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

#         # Perform convolution for each output channel
#         for k in range(out_channels):
#             for i in range(0, out_height * stride, stride):
#                 for j in range(0, out_width * stride, stride):
#                     # Extract the region from the padded input map for each channel
#                     input_region = input_map_padded[i:i + kernel_height, j:j + kernel_width, :]

#                     # Apply element-wise multiplication with the kernel and sum the results across channels
#                     output_map[i // stride, j // stride, k] = np.sum(input_region * kernel[:, :, k])

#         return output_map

#     return convolve

# # Example usage with three output channels for an RGB image
# input_map = np.random.randint(0, 255, size=(4, 4, 3))  # Random 4x4 RGB image
# kernel = np.random.randn(2, 2, 3)  # Random 2x2 kernel with 3 input channels and 3 output channels

# # Create a custom convolution function
# custom_conv2d = CustomConv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1, padding=1)

# # Perform convolution
# result = custom_conv2d(input_map, kernel)

# print("Input Map:")
# print(input_map)
# print("\nKernel:")
# print(kernel)
# print("\nConvolution Result:")
# print(result)


import numpy as np

t = 4



import torch
import torch.nn as nn
padding = 5

in_channels = 3

input_tensor = torch.randn((in_channels, 8, 8))
input_tensor = torch.randint(-8,9, size=(in_channels, 8, 8), dtype=torch.int8)
input_tensor = np.random.randint(-2,3, size=(3,4,8), dtype=int)

print(input_tensor.shape)
#print(torch.max(input_tensor[0]))
print(input_tensor[1])

input_tensor_padded = np.zeros((in_channels, input_tensor.shape[1]+2*padding,input_tensor.shape[2]+2*padding))
for c in range(in_channels):
    input_tensor_padded[c] = np.pad(input_tensor[c], ((padding, padding), (padding, padding)), mode='constant')

print(input_tensor_padded.shape)
#print(torch.max(input_tensor[0]))
print(input_tensor_padded[1])


34877673940.0

13083571383.0