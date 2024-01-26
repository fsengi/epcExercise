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
    approx_until = 0 #change this if u want to approximate the first bits by an approximate adder
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
    return MyNbitAdder(a+ 128,b+ 128)[0] - 256


def Mult_by_add(a,b):
    tmp = 0
    for i in range(b):
        tmp = MyWrapper(tmp,a)
    return tmp



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
                    accumulator = MyWrapper(accumulator, (roi[m, n] * kernel[m, n]))

            
            # Store the result in the output feature map
            output[i, j] = accumulator
    
    return output

# Example usage
image = np.array([[3, 3, 3],
                  [3, 3, 3],
                  [3, 3, 3]])

kernel = np.array([[2, 3],
                   [1, -1]])

result = convolution2d(image, kernel)
print("Original Image:")
print(image)
print("\nKernel:")
print(kernel)
print("\nResult of Convolution:")
print(result)


print(Mult_by_add(-4,24))