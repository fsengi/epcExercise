import os
import json
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import json 
import math

####################################
rows = 8
bit_list = range(0,rows)

algo_list = ["exact Serial [1]","Serial Aprox [2]", "SIAFA 1 [3]","SIAFA 2 [4]","SIAFA 3 [5]","SIAFA 4 [6]","exact Semi Serial [7]","Serial Aprox [8]", "exact parallel [9]","exact Semi Parallel [10]","own Aprox [11]", "own 3Memristors [12]"]

calcAllNewFlag = True 
demoDataFlag = False
###################################

# wood_img = io.imread("wood.jpg")
wood_img = io.imread("final_project/image_processing/wood.jpg")
print(wood_img.shape)

R_1 = wood_img[:, :, 0] 
G_1 = wood_img[:, :, 1]
B_1 = wood_img[:, :, 2]

#formula for converting colour(RGB) to Gray Image scale Image
Y_wood = (0.299 * np.array(R_1)) + (0.587 * np.array(G_1)) + (0.114 * np.array(B_1)) 

plt.imshow(Y_wood , cmap = "gray")
# print(Y_wood.shape)

# pattern for aproximation and exact adder
energy_consumption_list = []
energy_consumption_list.append([2153, 2043, 1941, 1881, 2069, 1976, 1966, 1892]) # exact Serial [1]
energy_consumption_list.append([772.01, 715.242, 716.736, 693.766, 751.802, 700.172, 688.285, 678.696]) # Serial Aprox [2]
energy_consumption_list.append([722.64, 801.437,667.417, 686.277, 759.353, 742.095, 695.799, 720.61]) # SIAFA 1 [3]
energy_consumption_list.append([1011.046, 955.49, 960.609, 946.953, 910.111, 898.936, 936.505, 927.341]) # SIAFA 2 [4]
energy_consumption_list.append([722.58, 709.08, 760.5, 744.12, 668.6, 689.24, 698.24, 724.26]) # SIAFA 3 [5]
energy_consumption_list.append([722.68, 709.38, 667.2, 687.02, 752.12, 729.72, 701.59, 721.4]) # SIAFA 4 [6]
energy_consumption_list.append([2446.8, 2518.8, 2284.5, 2313.1, 2374, 2436.9, 2292.3, 2326.9]) # exact Semi Serial [7]
energy_consumption_list.append([828.43, 817.43, 775.39, 838.65, 852.77, 843.23, 801.27, 852.65]) # Semi Serial Aprox [8]
energy_consumption_list.append([828.43, 817.43, 775.39, 838.65, 852.77, 843.23, 801.27, 852.65]) # exact Parallel [9]
energy_consumption_list.append([2068.7, 1962.8, 1853.7, 1797.8, 1982.9, 1893.6, 1882, 1811.2]) # exact Semi Parallel [10]
energy_consumption_list.append([696.69, 661.61, 641.48, 611.95, 642.26, 612.56, 581.57, 568.23]) # own Aprox [11]
energy_consumption_list.append([372.55, 316.45, 320.74, 308.92, 330.8, 279.65, 294.28, 243.13]) # own 3Memristors [12]


truthTable_s_list = []
truthTable_s_list.append([0, 1, 1, 0, 1, 0, 0, 1]) # exact Serial [1]
truthTable_s_list.append([1, 1, 1, 0, 1, 1, 0, 0]) # Serial Aprox [2]
truthTable_s_list.append([1, 1, 1, 0, 1, 1, 0, 0]) # SIAFA 1 [3]
truthTable_s_list.append([1, 1, 1, 0, 1, 0, 0, 0]) # SIAFA 2 [4]
truthTable_s_list.append([1, 1, 1, 1, 1, 0, 0, 0]) # SIAFA 3 [5]
truthTable_s_list.append([1, 1, 1, 0, 1, 0, 1, 0]) # SIAFA 4 [6]
truthTable_s_list.append([0, 1, 1, 0, 1, 0, 0, 1]) # exact Semi Serial [7]
truthTable_s_list.append([1, 1, 1, 0, 0, 0, 0, 0]) # Semi Serial Aprox [8]
truthTable_s_list.append([0, 1, 1, 0, 1, 0, 0, 1]) # exact parallel [9]
truthTable_s_list.append([0, 1, 1, 0, 1, 0, 0, 1]) # exact Semi Parallel [10]
truthTable_s_list.append([1, 1, 1, 1, 1, 1, 0, 1]) # own Aprox [11]
truthTable_s_list.append([1, 1, 1, 0, 1, 1, 1, 1]) # own 3Memristors [12]


truthTable_c_list = []
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 1, 1]) # exact Serial [1]
truthTable_c_list.append([0, 0, 0, 1, 0, 0, 1, 1]) # Serial Aprox [2]
truthTable_c_list.append([0, 0, 0, 1, 0, 0, 1, 1]) # SIAFA 1 [3]
truthTable_c_list.append([0, 1, 0, 1, 0, 1, 1, 1]) # SIAFA 2 [4]
truthTable_c_list.append([0, 0, 0, 0, 0, 1, 1, 1]) # SIAFA 3 [5]
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 0, 1]) # SIAFA 4 [6]
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 1, 1]) # exact Semi Serial [7]
truthTable_c_list.append([0, 0, 0, 1, 1, 1, 1, 1]) # Serial Aprox [8]
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 1, 1]) # exact parallel [9]
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 1, 1]) # exact Semi Parallel [10]
truthTable_c_list.append([0, 1, 0, 1, 0, 1, 1, 1]) # own Aprox [11]
truthTable_c_list.append([0, 1, 0, 1, 0, 1, 0, 1]) # own 3Memristors [12]

nameApprox_list = []
nameApprox_list.append("exact Serial [1]")
nameApprox_list.append("Serial Aprox [2]")
nameApprox_list.append("SIAFA 1 [3]")
nameApprox_list.append("SIAFA 2 [4]")
nameApprox_list.append("SIAFA 3 [5]")
nameApprox_list.append("SIAFA 4 [6]")
nameApprox_list.append("exact Semi Serial [7]")
nameApprox_list.append("Serial Aprox [8]")
nameApprox_list.append("exact parallel [9]")
nameApprox_list.append("exact Semi Parallel [10]")
nameApprox_list.append("own Aprox [11]")
nameApprox_list.append("own 3Memristors [12]")

blurrKernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
edgeDetectionKernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

kernel_list = []
kernel_list.append(blurrKernel)
kernel_list.append(edgeDetectionKernel)

kernelname_list = []
kernelname_list.append("blurring")
kernelname_list.append("edge Detection")

# write data to json file 

# Create a dictionary to hold the parsed data
loaded_dict = {}
empty_list = [0,0,0, 0,0,0, 0,0,0]

# Populate the dictionary
for i, name in enumerate(nameApprox_list):
    loaded_dict[name] = {"s": truthTable_s_list[i], "c": truthTable_c_list[i], "energy": energy_consumption_list[i], "ssi": empty_list, "psnr": empty_list, "energy_con": empty_list}

json_file_path = 'data.json'

# Write the data to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(loaded_dict, json_file, indent=4)

# assign the correct values for carry sum and energy acccording to choosen Algorithm
    
def Adder(a, b, c, approxAlgo = "exact Serial [1]"):
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

def My_Multiplier(a,b, approxAlgo, approxBit, blurrFlag):
    energy = 0
    res = 0
    if a < -1 and a > 1 and b < -1 and b > 1:

        a_wo_sign = abs(a)
        b_wo_sign = abs(b)

        if a_wo_sign > b_wo_sign:
            multcount = a
            multiplier = b
        else: 
            multcount = b
            multiplier = a

        if multcount < 0:
            multcount = multcount * (-1)
            multiplier = multiplier * (-1)
        
        for i in range(0, multcount):
            res, e  = MyNbitAdder(a=res, b=multiplier, Algo=approxAlgo, Bit=approxBit)
            energy += e
    else:
        res = a*b
    if blurrFlag == True:
        res = res >> 4
    return res, energy

def My_Mult(a, b, approxAlgo, approxBit, blurrFlag):
    '''go throw every pixel in 3x3 matrix'''
    energy = 0
    res = np.zeros((a.shape[0],a.shape[1]))
    for k in range(a.shape[0]):
        for l in range(a.shape[1]):
            res[k,l], e = My_Multiplier(a=int(a[k,l]), b=int(b[k,l]), approxAlgo=approxAlgo, approxBit=approxBit, blurrFlag=blurrFlag)
            # print(f'res:{res[k,l]}')
            energy += e
    return res, energy

# Convolution
def MyconvLUT(a, b, aproxAlgo, approxBit, blurrFlag, demoDataFlag):
    if demoDataFlag:
        return a, 1000000
    a = a.astype(int)
    b = b.astype(int)
    a_shape = np.shape(a)
    b_shape = np.shape(b)
    res_shape1 = np.abs(a_shape[0] - b_shape[0]) + 1
    res_shape2 = np.abs(a_shape[1] - b_shape[1]) + 1

    res = np.zeros((res_shape1, res_shape2))
    energy = 0
    for i in range(res_shape1):
        # print(f'row {i} of {res_shape1}')
        for j in range(res_shape2):
            resmatrix, ee = My_Mult(a=np.flip(b), b=a[i:i + b_shape[0], j:j + b_shape[1]], approxAlgo=aproxAlgo, approxBit=approxBit, blurrFlag=blurrFlag)                   
            res[i, j], e = MySum(matrix=resmatrix, approxAlgo=approxAlgo, approxBit=approxBit)
            energy += ee 
            energy += e
    return res, energy

def MySum(matrix, approxAlgo, approxBit):
    res = 0
    energy = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # print(f'{res}, {int(matrix[i,j])}, {approxAlgo}, {approxBit} ')
            res, e = MyNbitAdder(a=res, b=int(matrix[i,j]), Algo=approxAlgo, Bit=approxBit)
        energy += e
    return res, energy

#In 8 bit adder, lower 3 bits are implemented with approximate adder and rest of the with exact adder
def MyNbitAdder(a, b, Algo, Bit):
    try:
        #convert to binary and cut off the first two indices (they dont belong to the number but indicate that it is binary)
        minusABFlag = False

        if a >= 0 and b >= 0:    
            a_bin = bin(a)[2:]
            b_bin = bin(b)[2:]
        elif a < 0 and b >= 0:
            a_bin = decimal2TwoComplement(a)
            b_bin = bin(b)[2:]
        elif a >= 0 and b < 0:
            a_bin = bin(a)[2:]
            b_bin = decimal2TwoComplement(b)
        elif a < 0 and b < 0:
            minusABFlag = True
            a_bin = bin(a)[3:]
            b_bin = bin(b)[3:]
        else:
            print('Error')
        
        #reverse order of bytes for the adder
        rev_a , rev_b = list(a_bin[::-1]), list(b_bin[::-1])
        
        #We want to make the to bytes to equalt length such that we can add 
        #--> add zeros to the shortest list until it is the same as the longest
        if a >= 0 or minusABFlag: 
            rev_a = rev_a + max(0, len(rev_b)-len(rev_a)) * [0]
        else: 
            rev_a = rev_a + max(0, len(rev_b)-len(rev_a)) * [1]
        
        if b >= 0 or minusABFlag:
            rev_b = rev_b + max(0, len(rev_a)-len(rev_b)) * [0]
        else:
            rev_b = rev_b + max(0, len(rev_a)-len(rev_b)) * [1]

        carry_over  = 0
        total_sum   = 0
        total_energy = 0
        #############################################
        approx_until = Bit #change this if u want to approximate the first bits by an approximate adder
        #############################################
        sum_list = []
        #we want to do a bitwise addition
        for index, (bit1, bit2) in enumerate(zip(rev_a, rev_b) ):
            if index <= approx_until:
                #use approx_adder
                sum_element, carry_over, energy = Adder(int(bit1), int(bit2), int(carry_over), Algo) 
            else:
                #use exact_adder
                sum_element, carry_over, energy = Adder(int(bit1), int(bit2), int(carry_over))
            
            sum_list.append(int(sum_element))
            total_energy += energy
        
            
        if a+b < 0 and not minusABFlag:
            total_sum = twoComplement2Decimal(sum_list)
        elif a+b >= 0 and a >= 0 and b >= 0:
            sum_list.append(int(carry_over))
            total_sum = binary2Decimal(sum_list)
        elif a+b >= 0 and (a < 0 or b < 0):
            total_sum = binary2Decimal(sum_list)
        elif minusABFlag:
            sum_list.append(int(carry_over))
            total_sum = binary2Decimal(sum_list)
            total_sum = total_sum * (-1)

        return total_sum, total_energy #total energy in pJ!
    except Exception as e:
        print(f'Error: {e}')

def decimal2TwoComplement(num) -> list:
    num_bits = int(math.log2(abs(num))) + 3
    binary = bin(num & (2 ** num_bits - 1))[2:]  # Calculate two's complement
    bitlist = [int(bit) for bit in binary.zfill(num_bits)]
    # print(bitlist)
    return bitlist 

def twoComplement2Decimal(bitlist):
    # Check if the number is negative
    revnum = list(bitlist[::-1])

    # Perform two's complement negation
    flipped_bits = ''
    for bit in revnum:
        if bit == 0:
            flipped_bits = flipped_bits + '1'
        else: 
            flipped_bits = flipped_bits + '0'
    # print(flipped_bits)
    positive_decimal = int(flipped_bits, 2) + 1
    return -positive_decimal

def binary2Decimal(bitlist):
    # Convert the bit list to a string
    revnum = list(bitlist[::-1])
    bitstring = ''
    for bit in revnum:
        if bit == 1:
            bitstring = bitstring + '1'
        else: 
            bitstring = bitstring + '0'
    
    # Convert the bit string to a decimal number
    decimal_number = int(bitstring, 2)
    return decimal_number

add_approx_list = []
total_energy_lsit = []

def checkFilePresent(name):
    if os.path.isfile(f'{name}.png'):
        return True
    else: 
        return False

for kernel, kernel_name in zip(kernel_list, kernelname_list):
    # loop throw all Bitpositions 
    for approxAlgo in algo_list:
        # loop throw all Algorithm
        for approxBit in bit_list:
            approx_pic = 0
            approx_pic_x = 0
            approx_pic_y = 0
            print(f'kernel: {kernel_name} Algo: {approxAlgo} Bit: {approxBit}')
            if not calcAllNewFlag:
                if checkFilePresent(f'data_{kernel_name}/outputimage_{approxAlgo}_{approxBit}'):
                # if checkFilePresent(f'data/outputimage_{approxAlgo}_{approxBit}'):
                    continue
            if kernel_name == "blurring":
                blurrFlag = True
            else:
                blurrFlag = False
           
            approx_pic, total_energy = MyconvLUT(a=Y_wood, 
                                                 b=kernel, 
                                                 aproxAlgo=approxAlgo, 
                                                 approxBit=approxBit, 
                                                 demoDataFlag=demoDataFlag, 
                                                 blurrFlag=blurrFlag)

            np.save(f'data_{kernel_name}/outputimage_{approxAlgo}_{approxBit}.npy', approx_pic)
            plt.imsave(f'data_{kernel_name}/outputimage_{approxAlgo}_{approxBit}.png', approx_pic, cmap='gray')
            # plt.imsave(f'data/outputimage_{approxAlgo}_{approxBit}.png', approx_pic, cmap='gray')

            with open(f'data_{kernel_name}/{approxAlgo}_{approxBit}.json', 'w') as json_file:
            # with open(f'data/{approxAlgo}_{approxBit}.json', 'w') as json_file:
                json.dump(total_energy, json_file, indent=4)

