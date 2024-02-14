from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from scipy import signal
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import json 


wood_img = io.imread("wood.jpg")
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
energy_consumption_list.append([2068.7, 1962.8, 1853.7, 1797.8, 1982.9, 1893.6, 1882, 1811.2]) # exact adder
energy_consumption_list.append([696.69, 661.61, 641.48, 611.95, 642.26, 612.56, 581.57, 568.23]) # own Aprox
energy_consumption_list.append([722.64, 801.437,667.417, 686.277, 759.353, 742.095, 695.799, 720.61]) # SIAFA 1
energy_consumption_list.append([1011.046, 955.49, 960.609, 946.953, 910.111, 898.936, 936.505, 927.341]) # SIAFA 2
energy_consumption_list.append([722.58, 709.08, 760.5, 744.12, 668.6, 689.24, 698.24, 724.26]) # SIAFA 3
energy_consumption_list.append([722.68, 709.38, 667.2, 687.02, 752.12, 729.72, 701.59, 721.4]) # SIAFA 4
energy_consumption_list.append([696.69, 661.61, 641.48, 611.95, 642.26, 612.56, 581.57, 568.23]) # TODO Serial Aprox data missing
energy_consumption_list.append([696.69, 661.61, 641.48, 611.95, 642.26, 612.56, 581.57, 568.23]) # TODO Semi Serial Aprox data missing
energy_consumption_list.append([696.69, 661.61, 641.48, 611.95, 642.26, 612.56, 581.57, 568.23]) # TODO own 3Memristors


truthTable_s_list = []
truthTable_s_list.append([0, 1, 1, 0, 1, 0, 0, 1]) # exact adder
truthTable_s_list.append([1, 1, 1, 1, 1, 1, 0, 1]) # own Aprox
truthTable_s_list.append([1, 1, 1, 0, 1, 1, 0, 0]) # SIAFA 1
truthTable_s_list.append([1, 1, 1, 0, 1, 0, 0, 0]) # SIAFA 2
truthTable_s_list.append([1, 1, 1, 1, 1, 0, 0, 0]) # SIAFA 3
truthTable_s_list.append([1, 1, 1, 0, 1, 0, 1, 0]) # SIAFA 4
truthTable_s_list.append([1, 1, 1, 0, 1, 1, 0, 0]) # Serial Aprox
truthTable_s_list.append([1, 1, 1, 0, 0, 0, 0, 0]) # Semi Serial Aprox
truthTable_s_list.append([1, 1, 1, 0, 1, 1, 1, 1]) # own 3Memristors


truthTable_c_list = []
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 1, 1]) # exact adder
truthTable_c_list.append([0, 1, 0, 1, 0, 1, 1, 1]) # own Aprox
truthTable_c_list.append([0, 0, 0, 1, 0, 0, 1, 1]) # SIAFA 1
truthTable_c_list.append([0, 1, 0, 1, 0, 1, 1, 1]) # SIAFA 2
truthTable_c_list.append([0, 0, 0, 0, 0, 1, 1, 1]) # SIAFA 3
truthTable_c_list.append([0, 0, 0, 1, 0, 1, 0, 1]) # SIAFA 4
truthTable_c_list.append([0, 0, 0, 1, 0, 0, 1, 1]) # Serial Aprox
truthTable_c_list.append([0, 0, 0, 1, 1, 1, 1, 1]) # Semi Serial Aprox
truthTable_c_list.append([0, 1, 0, 1, 0, 1, 0, 1]) # own 3Memristors

nameApprox_list = []
nameApprox_list.append("exact")
nameApprox_list.append("own_Aprox")
nameApprox_list.append("SIAFA 1")
nameApprox_list.append("SIAFA 2")
nameApprox_list.append("SIAFA 3")
nameApprox_list.append("SIAFA 4")
nameApprox_list.append("Serial Aprox")
nameApprox_list.append("Semi Serial Aprox")
nameApprox_list.append("own_3Memristors")

blurrKernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
edgeDetectionKernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

kernel_list = []
kernel_list.append(blurrKernel)
kernel_list.append(edgeDetectionKernel)

kernelname_list = []
kernelname_list.append("blurring")
kernelname_list.append("edge_Detection")

# write data to json file 

# Create a dictionary to hold the parsed data
loaded_dict = {}
empty_list = [0,0,0,0,0,0,0,0,0]

# Populate the dictionary
for i, name in enumerate(nameApprox_list):
    loaded_dict[name] = {"s": truthTable_s_list[i], "c": truthTable_c_list[i], "energy": energy_consumption_list[i], "ssi": empty_list, "psnr": empty_list, "energy_con": empty_list}

json_file_path = 'data.json'

# Write the data to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(loaded_dict, json_file, indent=4)

# assign the correct values for carry sum and energy acccording to choosen Algorithm
    
def Adder(a, b, c, approxAlgo = "exact"):
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

def My_Multiplier(a,b, approxAlgo, approxBit, blurrFlag=False):
    energy = 0
    if a > b:
        res = a
        multiplier = a
        multcount = b
    else:
        res = b
        multiplier = b
        multcount = a
    
    for i in range(1, multcount):
        res, e  = MyNbitAdder(res, multiplier, approxAlgo, approxBit)
        energy += e
    # if blurrFlag:
    #     res = res >> 3
    return res, energy


def My_Mult(a, b, approxAlgo, approxBit, blurrFlag=False):
    '''go throw every pixel in 3x3 matrix'''
    energy = 0
    res = np.zeros((a.shape[0],a.shape[1]))
    for k in range(a.shape[0]):
        for l in range(a.shape[1]):
            res[k,l], e = My_Multiplier(int(a[k,l]), int(b[k,l]), approxAlgo, approxBit, blurrFlag)
            energy += e
    return res, energy

# Convolution

def MyconvLUT(a, b, aproxAlgo, aproxBit, blurrFlag=False, demoDataFlag=False):
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
        for j in range(res_shape2):
            resmatrix, energy = My_Mult(np.flip(b), a[i:i + b_shape[0], j:j + b_shape[1]], aproxAlgo, aproxBit, blurrFlag=blurrFlag)                   
            res[i, j], e = MySum(resmatrix, energy, aproxAlgo, aproxBit)
            energy += e
    return res, energy

def MySum(matrix, energy, approxAlgo, approxBit):
    res = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            res, e = MyNbitAdder(res, int(matrix[i,j]), approxAlgo, approxBit)
        energy += e
    return res, energy

#In 8 bit adder, lower 3 bits are implemented with approximate adder and rest of the with exact adder
def MyNbitAdder(a, b, Algo, Bit):
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

import os
import json

rows = 12
coll = 6
bit_list = range(0,rows)
algo_list = ["own_Aprox","SIAFA 1","SIAFA 2","SIAFA 3","SIAFA 4","Serial Aprox","Semi Serial Aprox","own_3Memristors"]
# algo_list = ["own_3Memristors"]

calcAllNewFlag = False 
demoDataFlag = False

add_approx_list = []
total_energy_lsit = []

def checkFilePresent(name):
    if os.path.isfile(f'{name}.png'):
        return True
    else: 
        return False

for kernel, kernel_name in zip(kernel_list, kernelname_list):
    blurrFlag = False
    # loop throw all Bitpositions 
    for indexAlgo, approxAlgo in enumerate(algo_list):
        # loop throw all Algorithm
        for indexBit, approxBit in enumerate(bit_list):
            print(f'kernel: {kernel_name} Algo: {approxAlgo} Bit: {approxBit}')
            if not calcAllNewFlag:
                # if checkFilePresent(f'data_{kernel_name}/outputimage_{approxAlgo}_{indexBit}'):
                if checkFilePresent(f'data/outputimage_{approxAlgo}_{indexBit}'):
                    continue
            if kernel_name == "blurring":
                blurrFlag = True
            # run calculation
            approx_pic, total_energy = MyconvLUT(Y_wood, kernel, approxAlgo, approxBit, demoDataFlag=demoDataFlag, blurrFlag=blurrFlag)
            # add_approx, max_Nbit_adder, total_energy = MyAdder(Y_einstein,Y_cap,approxAlgo,approxBit)

            # plt.imsave(f'data_{kernel_name}/outputimage_{approxAlgo}_{indexBit}.png', approx_pic, cmap='gray')
            plt.imsave(f'data/outputimage_{approxAlgo}_{indexBit}.png', approx_pic, cmap='gray')

            # with open(f'data_{kernel_name}/{approxAlgo}_{indexBit}.json', 'w') as json_file:
            with open(f'data/{approxAlgo}_{indexBit}.json', 'w') as json_file:
                json.dump(total_energy, json_file, indent=4)

        
