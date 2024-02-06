import numpy as np

def ExactAdder(a, b, c):
    if a==0 and b==0 and c==0:
        s=0
        c_out=0
    elif a==0 and b==0 and c==1:
        s=1
        c_out=0
    elif a==0 and b==1 and c==0:
        s=1
        c_out=0
    elif a==0 and b==1 and c==1:
        s=0
        c_out=1
    elif a==1 and b==0 and c==0:
        s=1
        c_out=0
    elif a==1 and b==0 and c==1:
        s=0
        c_out=1
    elif a==1 and b==1 and c==0:
        s=0
        c_out=1
    elif a==1 and b==1 and c==1:
        s=1
        c_out=1
    return s, c_out



# Sum logic fails for input combinations (0, 0, 0) and (1, 1, 1) 
def ApproxAdder(a, b, c):
    if a==0 and b==0 and c==0:
        s=1
        c_out=0
    elif a==0 and b==0 and c==1:
        s=1
        c_out=0
    elif a==0 and b==1 and c==0:
        s=1
        c_out=0
    elif a==0 and b==1 and c==1:
        s=0
        c_out=1
    elif a==1 and b==0 and c==0:
        s=1
        c_out=0
    elif a==1 and b==0 and c==1:
        s=0
        c_out=1
    elif a==1 and b==1 and c==0:
        s=0
        c_out=1
    elif a==1 and b==1 and c==1:
        s=0
        c_out=1
    return s, c_out


#In 8 bit adder, lower 3 bits are implemented with approximate adder and rest of the with exact adder
def MyNbitAdder(a,b):
    #convert to binary and cut off the first two indices (they dont belong to the number but indicate that it is binary)
    a_bin, b_bin = bin(a)[2:] , bin(b)[2:]
    
    #reverse order of bytes for the adder
    rev_a , rev_b = list(a_bin[::-1]), list(b_bin[::-1])
    
    
    #We want to make the to bytes to equalt length such that we can add 
    #--> add zeros to the shortest list until it is the same as the longest
    rev_a = rev_a + max(0, len(rev_b)-len(rev_a)) * [0]
    rev_b = rev_b + max(0, len(rev_a)-len(rev_b)) * [0]
    
    
    carry_over  = 0
    total_sum   = 0
    
    #############################################
    approx_until = 4 #change this if u want to approximate the first bits by an approximate adder
    #############################################

    #we want to do a bitwise addition
    count = 0
    for index, (bit1, bit2) in enumerate( zip(rev_a, rev_b) ):
        if index < approx_until:
            #use approx_adder
            sum_element, carry_over = ApproxAdder(int(bit1), int(bit2), int(carry_over) ) 
            count = count + 1
        else:
            #use exact_adder
            sum_element, carry_over = ExactAdder(int(bit1), int(bit2), int(carry_over) )
            count = count + 1
        total_sum += pow(2,index)*sum_element

    total_sum += pow(2,index+1)*carry_over
    return total_sum, count

print(MyNbitAdder(512,512))

def MyWrapper(a,b):
    if (((a + b) < 127) and ((a + b) > -128)):
        return MyNbitAdder(a+ 128,b+ 128)[0] - 256
    else:
        return 127


def Mult_by_add(a,b):
    tmp = 0
    for i in range(np.abs(b)):
        tmp = MyWrapper(tmp,a)
    if b > 0:
        return tmp
    elif b < 0:
        return -tmp
    else:
        return 0
    



# Function that addes two images
#max_Nbit_adder = np.zeros(Y_einstein.shape) ## maximum bit of adders required to add two decimal values
def MyAdder(f,g):
    f=np.array(f).astype(int)
    g=np.array(g).astype(int)
    #ensure that the size of the image is the same    
    res = np.zeros(f.shape)    
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):         
            #we will use a custom adding function
            res[i, j], max_Nbit_adder[i, j] = MyNbitAdder(f[i,j],g[i,j])
    return res, max_Nbit_adder

def convolution2d(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize the output feature map
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest (ROI) from the image
            roi = image[i:i+kernel_height, j:j+kernel_width]
            
            # Initialize the accumulator for the current position in the output
            accumulator = 0
            
            # Perform element-wise multiplication and accumulate
            for m in range(kernel_height):
                for n in range(kernel_width):
                    #accumulator += roi[m, n] * kernel[m, n]
                    #accumulator = accumulator + (roi[m, n] * kernel[m, n])
                    #accumulator = np.add(accumulator, (roi[m, n] * kernel[m, n]))
                    #accumulator = MyWrapper(accumulator, (roi[m, n] * kernel[m, n]))
                    accumulator = MyWrapper(accumulator, Mult_by_add(roi[m, n], kernel[m, n]))

            
            # Store the result in the output feature map
            # if (accumulator < 64):
            #     output[i, j] = accumulator
            # else:
            #     output[i,j] = 63
            output[i,j] = accumulator #int(accumulator/2)
    
    return output

def convolution2dpadding(image, kernel, padding):

    image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    print(np.shape(image))


    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the output dimensions
    output_height = (image_height - kernel_height) + 1
    output_width = (image_width - kernel_width) + 1
    print(output_height)

    # # Pad image
    # tmp = np.zeros((output_height, output_width), dtype = int)
    # for i in range(output_height)[padding:-padding]:
    #     for j in range(output_width)[padding:-padding]:
    #         tmp[i,j] = tmp[i,j] = image[i-padding, j-padding]

    
    # Initialize the output feature map
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest (ROI) from the image
            roi = image[i:i+kernel_height, j:j+kernel_width]
            
            # Initialize the accumulator for the current position in the output
            accumulator = 0
            
            # Perform element-wise multiplication and accumulate
            for m in range(kernel_height):
                for n in range(kernel_width):
                    #accumulator += roi[m, n] * kernel[m, n]
                    #accumulator = accumulator + (roi[m, n] * kernel[m, n])
                    #accumulator = np.add(accumulator, (roi[m, n] * kernel[m, n]))
                    #accumulator = MyWrapper(accumulator, (roi[m, n] * kernel[m, n]))
                    accumulator = MyWrapper(accumulator, Mult_by_add(roi[m, n], kernel[m, n]))

            
            # Store the result in the output feature map
            # if (accumulator < 64):
            #     output[i, j] = accumulator
            # else:
            #     output[i,j] = 63
            output[i,j] = accumulator #int(accumulator/2)
    return output




def convolution2dpaddingstride(image, kernel, stride, padding):

    #stride to bitshift
    if stride == 2:
        shift = 1
    else:
        shift = 0

    # Pad image
    image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the output dimensions using bitshifts
    output_height = ((image_height - kernel_height) >> shift) + 1
    output_width = ((image_width - kernel_width) >> shift )+ 1

    # Initialize the output feature map
    output = np.zeros((output_height, output_width))


    # Perform the convolution with stride
    for i in range(0, output_height * stride, stride):
        for j in range(0, output_width * stride, stride):
            # Extract the region of interest (ROI) from the image
            roi = image[i:i+kernel_height, j:j+kernel_width]

            # Initialize the accumulator for the current position in the output
            accumulator = 0

            # Perform element-wise multiplication and accumulate
            for m in range(kernel_height):
                for n in range(kernel_width):
                    #accumulator += roi[m, n] * kernel[m, n]
                    accumulator = MyWrapper(accumulator, Mult_by_add(roi[m, n], kernel[m, n]))

            # Assign the accumulated value to the output
            output[(i >> shift), (j >> shift)] = accumulator

    return output


def varchannel_conv(image, kernel, stride, padding):

    #stride to bitshift
    if stride == 2:
        shift = 1
    else:
        shift = 0

    #Determine number of in_channels
    in_channels = image.shape[0]

    # Apply zero-padding to the input image on all in_channels
    image_padded = np.zeros((in_channels, image.shape[1]+2*padding,image.shape[2]+2*padding), dtype=int)
    for c in range(in_channels):
        image_padded[c] = np.pad(image[c], ((padding, padding), (padding, padding)), mode='constant')

    print(image_padded[0])

    # Get dimensions of the padded image and kernel
    image_height, image_width = image_padded.shape[1], image_padded.shape[2]
    kernel_height, kernel_width = kernel.shape[1], kernel.shape[2]

    # Calculate the output dimensions using bitshifts
    output_height = ((image_height - kernel_height) >> shift) + 1
    output_width = ((image_width - kernel_width) >> shift )+ 1
    output_depth = kernel.shape[0]
    output_channels = output_depth

    # Initialize the output feature map
    output = np.zeros((output_depth, output_height, output_width), dtype=int)


    # Perform the convolution with stride
    for i in range(0, output_height * stride, stride):
        for j in range(0, output_width * stride, stride):
            # Extract the region of interest (ROI) from the image
            roi = image_padded[:, i:i+kernel_height, j:j+kernel_width]

            # Initialize the accumulator for the current position in the output
            accumulator = 0

            # Perform element-wise multiplication and accumulate
            for k in range(output_channels):
                for m in range(kernel_height):
                    for n in range(kernel_width):
                        for c in range(in_channels):
                            #accumulator += roi[m, n] * kernel[m, n]
                            accumulator = MyWrapper(accumulator, Mult_by_add(roi[c, m, n], kernel[k, m, n]))

                # Assign the accumulated value to the output
                output[k, (i >> shift), (j >> shift)] = accumulator

    return output


def multichannelconv(in_channels, out_channels, image, kernel, stride, padding):

    #stride to bitshift
    if stride == 2:
        shift = 1
    else:
        shift = 0

    # print(shift)

    # Get dimensions of the image and kernel
    image_height, image_width = image[0].shape 
    image_height += 2*padding
    image_width += 2*padding
    kernel_height, kernel_width = kernel[0].shape
    # print(image[0].shape)
    # print(kernel[0].shape)
    # Calculate the output dimensions using bitshifts
    output_height = ((image_height - kernel_height) >> shift) + 1
    output_width = ((image_width - kernel_width) >> shift )+ 1

    output_map = np.zeros(( out_channels, output_height, output_width))

    for channel in range(in_channels):
        # print(output_map[channel,:,:])
        # print(image[channel,:,:])
        # print(kernel[channel,:,:])
        # print(convolution2dpaddingstride(image[channel], kernel[channel], stride, padding))
        output_map[channel] = convolution2dpaddingstride(image[channel], kernel[channel], stride, padding)

    return output_map




def conv2d(image, kernel, stride=1, padding=0, bias=None):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the output dimensions
    output_height = (image_height - kernel_height + 2 * padding) // stride + 1
    output_width = (image_width - kernel_width + 2 * padding) // stride + 1
    
    # Initialize the output feature map
    output = np.zeros((output_height, output_width))
    

    
    # Perform the convolution with stride
    for i in range(0, output_height, stride):
        for j in range(0, output_width, stride):
            # Extract the region of interest (ROI) from the padded image
            roi = image_padded[i:i+kernel_height, j:j+kernel_width]

            # Initialize the accumulator for the current position in the output
            accumulator = 0
            
            # Perform element-wise multiplication and accumulate
            for m in range(kernel_height):
                for n in range(kernel_width):
                    #accumulator += roi[m, n] * kernel[m, n]
                    #accumulator = accumulator + (roi[m, n] * kernel[m, n])
                    #accumulator = np.add(accumulator, (roi[m, n] * kernel[m, n]))
                    #accumulator = MyWrapper(accumulator, (roi[m, n] * kernel[m, n]))
                    accumulator = MyWrapper(accumulator, Mult_by_add(roi[m, n], kernel[m, n]))
            
            # Add bias if provided
            if bias is not None:
                accumulator += bias
            
            # Store the result in the output feature map
            # print(accumulator)
            # print(type(accumulator))
            # print(accumulator[0])
            output[i//stride, j//stride] = accumulator
    
    return output

# # Example usage
# image = np.array([[3, 3, 3],
#                   [3, 3, 3],
#                   [3, 3, 3]])

# kernel = np.array([[5, 3],
#                    [1, -1]])

# result = conv2d(image, kernel)
# print("Original Image:")
# print(image)
# print("\nKernel:")
# print(kernel)
# print("\nResult of Convolution:")
# print(result)
# h = -100
# t = -1
# print(h * t)
# print(Mult_by_add(t,h))

def max_pooling(input_array, pool_size=(2, 2)):
    """
    Applies max pooling to a 2D array.

    Parameters:
    - input_array: 2D numpy array
    - pool_size: Tuple of two integers, specifying the size of the pooling window

    Returns:
    - 2D numpy array after max pooling
    """
    if len(input_array.shape) != 2:
        raise ValueError("Input array must be 2D")

    rows, cols = input_array.shape
    pool_rows, pool_cols = pool_size

    # Calculate the output shape after pooling
    out_rows = rows // pool_rows
    out_cols = cols // pool_cols

    # Reshape the input array to facilitate pooling
    reshaped_array = input_array[:out_rows * pool_rows, :out_cols * pool_cols].reshape(
        out_rows, pool_rows, out_cols, pool_cols
    )

    # Apply max pooling along the specified axis
    pooled_array = reshaped_array.max(axis=(1, 3))

    return pooled_array


def relu(x):
    res = np.zeros((np.shape(x)[0], np.shape(x)[1]), dtype=int)
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            res[i,j] = np.maximum(0,x[i,j])
    return res

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def initialize_weights(shape):
    return np.random.randint(-1, 2, size=shape, dtype=int)

def initialize_bias(shape):
    return np.zeros(shape, dtype=int)

def simple_convolutional_net(input_image):
    # Define the architecture
    W1 = initialize_weights((3, 3))
    W2 = initialize_weights((3, 3))
    W3 = initialize_weights((3, 3))
    W4 = initialize_weights((3, 3))
    
    # Forward pass
    #conv1 = convolution2d(input_image, W1)
    conv1 = convolution2dpadding(input_image, W1, 2)
    relu1 = relu(conv1)
    # print(input_image)
    print(np.max(input_image))
    # print(relu1)
    print(np.shape(relu1))
    print(np.max(relu1))

    relu1 = max_pooling(relu1)
    
    conv2 = convolution2d(relu1,W2)
    relu2 = relu(conv2)


    # print(relu2)
    print(np.shape(relu2))
    print(np.max(relu2))

    relu2 = max_pooling(relu2)
    
    conv3 = convolution2d(relu2,W3)
    relu3 = relu(conv3)

    # print(relu2)
    print(np.shape(relu3))
    print(np.max(relu3))
    
    conv4 = convolution2d(relu3,W4)
    relu4 = relu(conv4)
    
    # Flatten the output for fully connected layer
    flattened_output = relu4.flatten()
    
    # Fully connected layer
    fc_weights = initialize_weights((flattened_output.shape[0], 10))
    fc_output = np.dot(flattened_output, fc_weights)
    
    # Apply softmax for classification
    output_probabilities = softmax(fc_output)
    
    return output_probabilities

#Example usage
input_image = np.random.randint(-4, 3, size=(100,100))
output_probabilities = simple_convolutional_net(input_image)
print("Output Probabilities:", output_probabilities)

# imi = initialize_weights((8,8))
# Wt = initialize_weights((3,3))
# print(Wt)
# print(np.shape(Wt))


import torch
import torch.nn as nn

# print(imi)
# print(np.shape(imi))
# residi = convolution2d(imi, Wt)
# print(residi)
# print(np.shape(residi))
# residi0 = relu(residi)
# print(residi0)
# print(np.shape(residi0))
# residi1 = convolution2d(residi0, Wt)
# print(residi1)
# print(np.shape(residi1))
# residi2 = relu(residi1)
# print(residi2)
# print(np.shape(residi2))

# output_height = 10  # Replace 10 with the actual value of output_height
# N = 2  # Replace 2 with the number of elements you want to exclude from the beginning and end

# selected_elements = list(range(output_height))[N:-N]

# print(selected_elements)

# output_height = 12
# output_width = 12
padding = 1
stride = 1
kernel = np.random.randint(-2, 2, size=(1,9,9), dtype=int)
#kernel = torch.randint(-1,2, size=(3, 2, 2), dtype=torch.int8)
# print(kernel)
#kernel = np.array([[2,0,0],
                #    [0,2,0],
                #    [1,0,2]], dtype=int)
#image = np.random.randint(-4, 4, size=(8,8))
image = np.random.randint(-4,5, size=(3,16,16), dtype=int)
#image = torch.randint(-8,9, size=(3, 8, 8), dtype=torch.int8)
# print(kernel)
# print(np.shape(image))
# print(image[0,:,:])
# ref = convolution2dpadding(image, kernel, padding)
# rusi = convolution2dpaddingstride(image, kernel, padding, stride)
# print(np.shape(ref))
# print(ref)
# print(np.shape(rusi))
# print(rusi)

todo = varchannel_conv(image, kernel, stride, padding)

print(todo.shape)
print(todo)





# tmp = np.zeros((output_height, output_width))
# for i in range(output_height)[padding:-padding]:
#     for j in range(output_width)[padding:-padding]:
#         tmp[i,j] = image[i-padding, j-padding]

# print(tmp)