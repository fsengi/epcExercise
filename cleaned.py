import numpy as np


# pattern for aproximation and exact adder
energy_consumption_list = []
energy_consumption_list.append([2068.7, 1962.8, 1853.7, 1797.8, 1982.9, 1893.6, 1882, 1811.2]) # exact adder
energy_consumption_list.append([696.69, 661.61, 641.48, 611.95, 642.26, 612.56, 581.57, 568.23]) # own Aprox
energy_consumption_list.append([722.64, 801.437,667.417, 686.277, 759.353, 742.095, 695.799, 720.61]) # SIAFA 1
energy_consumption_list.append([1011.046, 955.49, 960.609, 946.953, 910.111, 898.936, 936.505, 927.341]) # SIAFA 2
energy_consumption_list.append([722.58, 709.08, 760.5, 744.12, 668.6, 689.24, 698.24, 724.26]) # SIAFA 3
energy_consumption_list.append([722.68, 709.38, 667.2, 687.02, 752.12, 729.72, 701.59, 721.4]) # SIAFA 4
energy_consumption_list.append([696.69, 661.61, 641.48, 611.95, 642.26, 612.56, 581.57, 568.23]) # TODO Serial Aprox data missing
energy_consumption_list.append([696.69, 661.61, 641.48, 611.95, 642.26, 612.56, 581.57, 568.23]) # TODO Semi Serial Aprox data missing

truthTable_s_list = []
truthTable_s_list.append([0, 1, 1, 0, 1, 0, 0, 1]) # exact adder
truthTable_s_list.append([1, 1, 1, 1, 1, 1, 0, 1]) # own Aprox
truthTable_s_list.append([1, 1, 1, 0, 1, 1, 0, 0]) # SIAFA 1
truthTable_s_list.append([1, 1, 1, 0, 1, 0, 0, 0]) # SIAFA 2
truthTable_s_list.append([1, 1, 1, 1, 1, 0, 0, 0]) # SIAFA 3
truthTable_s_list.append([1, 1, 1, 0, 1, 0, 1, 0]) # SIAFA 4
truthTable_s_list.append([1, 1, 1, 0, 1, 1, 0, 0]) # Serial Aprox
truthTable_s_list.append([1, 1, 1, 0, 0, 0, 0, 0]) # Semi Serial Aprox


truthTable_c_list = []
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 1, 1]) # exact adder
truthTable_c_list.append([0, 1, 0, 1, 0, 1, 1, 1]) # own Aprox
truthTable_c_list.append([0, 0, 0, 1, 0, 0, 1, 1]) # SIAFA 1
truthTable_c_list.append([0, 1, 0, 1, 0, 1, 1, 1]) # SIAFA 2
truthTable_c_list.append([0, 0, 0, 0, 0, 1, 1, 1]) # SIAFA 3
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 0, 1]) # SIAFA 4
truthTable_c_list.append([0, 0, 0, 1, 0, 0, 1, 1]) # Serial Aprox
truthTable_c_list.append([0, 0, 0, 1, 1, 1, 1, 1]) # Semi Serial Aprox


nameApprox_list = []
nameApprox_list.append("exact")
nameApprox_list.append("own_Aprox")
# nameApprox_list.append("SIAFA 1")
# nameApprox_list.append("SIAFA 2")
# nameApprox_list.append("SIAFA 3")
# nameApprox_list.append("SIAFA 4")
# nameApprox_list.append("Serial Aprox")
# nameApprox_list.append("Semi Serial Aprox")



# write data to json file 
import json 
# Create a dictionary to hold the parsed data
parsed_data = {}
empty_list = [0,0,0,0,0,0,0,0,0]

# Populate the dictionary
for i, name in enumerate(nameApprox_list):
    parsed_data[name] = {"s": truthTable_s_list[i], "c": truthTable_c_list[i], "energy": energy_consumption_list[i], "ssi": empty_list, "psnr": empty_list, "energy_con": empty_list}

json_file_path = 'data.json'

# Write the data to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(parsed_data, json_file, indent=4)


def Adder(a, b, c, approxAlgo = 'exact'):
    if a==0 and b==0 and c==0:
        s = loaded_dict[approxAlgo]["s"][0]
        c_out = loaded_dict[approxAlgo]["c"][0]
        energy_consumption = loaded_dict[approxAlgo]["energy"][0]
    elif a==0 and b==0 and c==1:
        s = loaded_dict[approxAlgo]["s"][1]
        c_out = loaded_dict[approxAlgo]["c"][1]
        energy_consumption = loaded_dict[approxAlgo]["energy"][1]
    elif a==0 and b==1 and c==0:
        s = loaded_dict[approxAlgo]["s"][2]
        c_out = loaded_dict[approxAlgo]["c"][2]
        energy_consumption = loaded_dict[approxAlgo]["energy"][2]
    elif a==0 and b==1 and c==1:
        s = loaded_dict[approxAlgo]["s"][3]
        c_out = loaded_dict[approxAlgo]["c"][3]
        energy_consumption = loaded_dict[approxAlgo]["energy"][3]
    elif a==1 and b==0 and c==0:
        s = loaded_dict[approxAlgo]["s"][4]
        c_out = loaded_dict[approxAlgo]["c"][4]
        energy_consumption = loaded_dict[approxAlgo]["energy"][4]
    elif a==1 and b==0 and c==1:
        s = loaded_dict[approxAlgo]["s"][5]
        c_out = loaded_dict[approxAlgo]["c"][5]
        energy_consumption = loaded_dict[approxAlgo]["energy"][5]
    elif a==1 and b==1 and c==0:
        s = loaded_dict[approxAlgo]["s"][6]
        c_out = loaded_dict[approxAlgo]["c"][6]
        energy_consumption = loaded_dict[approxAlgo]["energy"][6]
    elif a==1 and b==1 and c==1:
        s = loaded_dict[approxAlgo]["s"][7]
        c_out = loaded_dict[approxAlgo]["c"][7]
        energy_consumption = loaded_dict[approxAlgo]["energy"][7]
    return s, c_out, energy_consumption



#In 8 bit adder, lower 3 bits are implemented with approximate adder and rest of the with exact adder
def SunnyMyNbitAdder(a, b, Algo, Bit):
    try:
        #convert to binary and cut off the first two indices (they dont belong to the number but indicate that it is binary)
        a_bin, b_bin = bin(a)[2:], bin(b)[2:]
        
        #reverse order of bytes for the adder
        rev_a , rev_b = list(a_bin[::-1]), list(b_bin[::-1])
        
        #We want to make the to bytes to equalt length such that we can add 
        #--> add zeros to the shortest list until it is the same as the longest
        rev_a = rev_a + max(0, len(rev_b)-len(rev_a)) * [0]
        rev_b = rev_b + max(0, len(rev_a)-len(rev_b)) * [0]

        carry_over  = 0
        total_sum   = 0
        
        #############################################
        approx_until = Bit #change this if u want to approximate the first bits by an approximate adder
        #############################################

        #we want to do a bitwise addition
        count = 0
        total_energy = 0
        for index, (bit1, bit2) in enumerate(zip(rev_a, rev_b) ):
            if index < approx_until:
                #use approx_adder
                sum_element, carry_over, _energy = Adder(int(bit1), int(bit2), int(carry_over), Algo) 
            else:
                #use exact_adder
                sum_element, carry_over, _energy = Adder(int(bit1), int(bit2), int(carry_over))
            
            count = count + 1
            total_energy += _energy

            total_sum += pow(2,index) * sum_element

        total_sum += pow(2,index+1) * carry_over
       
        return total_sum, total_energy #total energy in pJ!
    except Exception as e:
        print(f'Error: {e}')




# Load the data from the JSON file into a dictionary
with open(json_file_path, 'r') as json_file:
    loaded_dict = json.load(json_file)


def SunnyMyWrapper(a,b, alg, bit):
    global tot_enegery
    global num_steps
    tmp = 5*bit + 17*(8-bit)  # Our own algorithm takes 5 timesteps per one bit addition, the exact algorithm we are comparing to takes 17 timsteps
    num_steps += tmp
    if (((a + b) < 127) and ((a + b) > -128)):
        tot_enegery += SunnyMyNbitAdder(a+ 128,b+ 128,alg, bit)[1]
        return SunnyMyNbitAdder(a+ 128,b+ 128, alg, bit)[0] - 256
    else:
        tot_enegery += SunnyMyNbitAdder(a+ 128,b+ 128,alg, bit)[1]
        return 127

def sunMult_by_add(a,b):
    tmp = 0
    for i in range(np.abs(b)):
        tmp = SunnyMyWrapper(tmp,a, algorithm, bit)
    if b > 0:
        return tmp
    elif b < 0:
        return -tmp
    else:
        return 0



def sunvarchannel_conv(image, kernel, stride, padding):

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

    # print(image_padded[0])

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
                            accumulator = SunnyMyWrapper(accumulator, sunMult_by_add(roi[c, m, n], kernel[k, m, n]), algorithm, bit)

                # Assign the accumulated value to the output
                output[k, (i >> shift), (j >> shift)] = accumulator

    return output




def skipconnection(current, old):
    if np.max(old) != 0:
        tmp = ((current + old)/(np.max(old+ current)/100))
    else: 
        tmp = current + old
    return np.round(tmp).astype(int)


def relu(x):
    res = np.zeros((np.shape(x)[0], np.shape(x)[1], np.shape(x)[2]), dtype=int)
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            for k in range(np.shape(x)[2]):
                res[i,j,k] = np.maximum(0,x[i,j,k])
    return res

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(shape):
    return np.random.randint(-1, 2, size=shape, dtype=int)

def initialize_bias(shape):
    return np.zeros(shape, dtype=int)




def resnet(input_image):
    # algorithm = "own_Aprox"
    # bit = 4

    # Initialize Weights
    K0 = np.random.randint(-2, 2, size=(6,5,5), dtype=int)
    K1 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K2 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K3 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K4 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K5 = np.random.randint(-2, 2, size=(12,3,3), dtype=int)
    K6 = np.random.randint(-2, 2, size=(12,3,3), dtype=int)
    K7 = np.random.randint(-2, 2, size=(12,3,3), dtype=int)
    K8 = np.random.randint(-2, 2, size=(12,3,3), dtype=int)
    K9 = np.random.randint(-2, 2, size=(12,3,3), dtype=int)
    K10 = np.random.randint(-2, 2, size=(24,3,3), dtype=int)
    K11 = np.random.randint(-2, 2, size=(24,3,3), dtype=int)
    K12 = np.random.randint(-2, 2, size=(24,3,3), dtype=int)
    K13 = np.random.randint(-2, 2, size=(24,3,3), dtype=int)
    K14 = np.random.randint(-2, 2, size=(1,3,3), dtype=int)


    # Forward pass
    conv0 = sunvarchannel_conv(input_image, K0, 2, 0)
    relu0 = relu(conv0)
    print(relu0.shape)


    conv1 = sunvarchannel_conv(relu0, K1, 1, 1)
    print(conv1.shape)
    relu1 = relu(conv1)
    conv2 = sunvarchannel_conv(relu1, K2, 1, 1)
    print(conv2.shape)
    relu2 = relu(conv2)

    skip_connection0 = skipconnection(relu2, relu0)
    print(skip_connection0.shape)
    

    conv3 = sunvarchannel_conv(skip_connection0, K3, 1, 1)
    print(conv3.shape)
    relu3 = relu(conv3)
    conv4 = sunvarchannel_conv(relu3, K4, 1, 1)
    print(conv4.shape)
    relu4 = relu(conv4)

    skip_connection1 = skipconnection(relu4, skip_connection0)
    print(skip_connection1.shape)

    conv5 = sunvarchannel_conv(skip_connection1, K5, 2, 1)
    print(conv5.shape)
    relu5 = relu(conv5)


    conv6 = sunvarchannel_conv(relu5, K6, 1, 1)
    print(conv6.shape)
    relu6 = relu(conv6)
    conv7 = sunvarchannel_conv(relu6, K7, 1, 1)
    print(conv7.shape)
    relu7 = relu(conv7)

    skip_connection2 = skipconnection(relu5, relu7)
    print(skip_connection2.shape)
    

    conv8 = sunvarchannel_conv(skip_connection2, K8, 1, 1)
    print(conv8.shape)
    relu8 = relu(conv8)
    conv9 = sunvarchannel_conv(relu8, K9, 1, 1)
    print(conv9.shape)
    relu9 = relu(conv9)

    skip_connection3 = skipconnection(relu9, skip_connection2)
    print(skip_connection3.shape)


    conv10 = sunvarchannel_conv(skip_connection3, K10, 2, 0)
    print(conv10.shape)
    relu10 = relu(conv10)
    conv11 = sunvarchannel_conv(relu10, K11, 1, 1)
    print(conv11.shape)
    relu11 = relu(conv11)

    skip_connection4 = skipconnection(relu10, relu11)
    print(skip_connection4.shape)
    

    conv12 = sunvarchannel_conv(skip_connection4, K12, 1, 1)
    print(conv12.shape)
    relu12 = relu(conv12)
    conv13 = sunvarchannel_conv(relu12, K12, 1, 1)
    print(conv13.shape)
    relu13 = relu(conv13)

    skip_connection5 = skipconnection(relu13, skip_connection4)
    print(skip_connection4.shape)


    flattened_output = skip_connection5.flatten()
    print(flattened_output.shape)

    fc_weights = initialize_weights((flattened_output.shape[0], 10))
    fc_output = np.dot(flattened_output, fc_weights)
    
    # Apply softmax for classification
    output_probabilities = softmax(fc_output)
    
    return output_probabilities



def convolutional_net(input_image):

     # Initialize Weights
    K0 = np.random.randint(-2, 2, size=(6,5,5), dtype=int)
    K1 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K2 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K3 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K4 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K5 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K6 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K7 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K8 = np.random.randint(-2, 2, size=(6,3,3), dtype=int)
    K9 = np.random.randint(-2, 2, size=(32,3,3), dtype=int)
    K10 = np.random.randint(-2, 2, size=(32,3,3), dtype=int)
    K11 = np.random.randint(-2, 2, size=(32,3,3), dtype=int)    
    K12 = np.random.randint(-2, 2, size=(32,3,3), dtype=int)
    K13 = np.random.randint(-2, 2, size=(64,3,3), dtype=int)

    # Forward pass
    conv0 = sunvarchannel_conv(input_image, K0, 2, 0)
    relu0 = relu(conv0)
    print(relu0.shape)


    conv1 = sunvarchannel_conv(relu0, K1, 1, 0)
    print(conv1.shape)
    relu1 = relu(conv1)
    conv2 = sunvarchannel_conv(relu1, K2, 1, 0)
    print(conv2.shape)
    relu2 = relu(conv2)
    

    conv3 = sunvarchannel_conv(relu2, K3, 1, 0)
    print(conv3.shape)
    relu3 = relu(conv3)
    conv4 = sunvarchannel_conv(relu3, K4, 1, 0)
    print(conv4.shape)
    relu4 = relu(conv4)


    conv5 = sunvarchannel_conv(relu4, K5, 1, 1)
    print(conv5.shape)
    relu5 = relu(conv5)

    conv6 = sunvarchannel_conv(relu5, K6, 1, 1)
    print(conv6.shape)
    relu6 = relu(conv6)

    conv7 = sunvarchannel_conv(relu6, K7, 1, 1)
    print(conv7.shape)
    relu7 = relu(conv7)

    conv8 = sunvarchannel_conv(relu7, K8, 1, 1)
    print(conv8.shape)
    relu8 = relu(conv8)


    conv9 = sunvarchannel_conv(relu8, K9, 2, 1)
    print(conv9.shape)
    relu9 = relu(conv9)

    conv10 = sunvarchannel_conv(relu9, K10, 1, 1)
    print(conv10.shape)
    relu10 = relu(conv10)    

    conv11 = sunvarchannel_conv(relu10, K11, 1, 1)
    print(conv11.shape)
    relu11 = relu(conv11)

    conv12 = sunvarchannel_conv(relu11, K12, 1, 1)
    print(conv12.shape)
    relu12 = relu(conv12)

    conv13 = sunvarchannel_conv(relu12, K13, 1, 1)
    print(conv13.shape)
    relu13 = relu(conv13)

    flattened_output = relu13.flatten()
    print(flattened_output.shape)

    fc_weights = initialize_weights((flattened_output.shape[0], 10))
    fc_output = np.dot(flattened_output, fc_weights)
    
    # Apply softmax for classification
    output_probabilities = softmax(fc_output)

    return output_probabilities




image = np.random.randint(-5,5, size=(3,40,40), dtype=int)


# tot_enegery = 0
# prob = resnet(image)
# print(prob)
# print(tot_enegery)

# image = np.random.randint(-5,5, size=(3,52,52), dtype=int)

# tot_enegery = 0
# prob = convolutional_net(image)
# print(prob)
# print(tot_enegery)



# for i in range(9):
#     bit = i
#     tot_enegery = 0
#     resnet(image)
#     print(bit)
#     print(tot_enegery)






import csv

# Open a CSV file in write mode
with open('testout1.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write header row
    csv_writer.writerow(['Approx. Algo', 'Bit', 'Total Energy'])

    # Loop through the range
    for name in nameApprox_list:
        algorithm = name
        for i in range(9):
            try:
                bit = i
                tot_enegery = 0 
                num_steps = 0
                resnet(image)
                
                # Print bit and tot_energy
                print(bit)
                print(tot_enegery)
                print(name)

                # Write bit and tot_energy to the CSV file
                csv_writer.writerow([name, bit, tot_enegery, num_steps])
            except Exception as e:
                print('error at', name, bit)
                continue



with open('testout.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write header row
    csv_writer.writerow(['Approx. Algo', 'Bit', 'Total Energy'])

    # Loop through the range
    for name in nameApprox_list:
        algorithm = name
        for i in range(9):
            try:
                bit = i
                tot_enegery = 0 
                num_steps = 0
                convolutional_net(image)
                
                # Print bit and tot_energy
                print(bit)
                print(tot_enegery)
                print(name)

                # Write bit and tot_energy to the CSV file
                csv_writer.writerow([name, bit, tot_enegery, num_steps])
            except Exception as e:
                print('error at', name, bit)
                continue
