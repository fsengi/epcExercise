####################################
# this script is used to calculate all the pictures 
# it is based on lists, so new truthtables can be added and calculated fully automated on a server, big time saver
####################################
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io



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
energy_consumption_list.append([0, 0, 0, 0, 0, 0, 0, 0]) # C51 paper [13]


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
truthTable_s_list.append([1, 1, 1, 0, 0, 0, 0, 0]) # C51 paper [13]


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
truthTable_c_list.append([0, 0, 0, 1, 1, 1, 1, 1]) # C51 paper [13]


truthTable_steps_list = []
truthTable_steps_list.append(22) # exact Serial [1]
truthTable_steps_list.append(8) # Serial Aprox [2]
truthTable_steps_list.append(8) # SIAFA 1 [3]
truthTable_steps_list.append(10) # SIAFA 2 [4]
truthTable_steps_list.append(8) # SIAFA 3 [5]
truthTable_steps_list.append(8) # SIAFA 4 [6]
truthTable_steps_list.append(10) # exact Semi Serial [7]
truthTable_steps_list.append(5) # Semi Serial Aprox [8]
truthTable_steps_list.append(5) # exact parallel [9]
truthTable_steps_list.append(17) # exact Semi Parallel [10]
truthTable_steps_list.append(5) # own Aprox [11]
truthTable_steps_list.append(10) # C51 paper [13]

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
nameApprox_list.append("C51 paper [13]")

edgeDetectionKernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
blurrKernel = np.array([[1,2,1],[2,4,2],[1,2,1]])

kernel_list = []
kernel_list.append(edgeDetectionKernel)
kernel_list.append(blurrKernel)

kernelname_list = []
kernelname_list.append("edge Detection")
kernelname_list.append("blurring")

# write data to json file
# Create a dictionary to hold the parsed data
loaded_dict = {}
results_dict = {}

# Populate the dictionary for getting truth table data
for i, name in enumerate(nameApprox_list):
    loaded_dict[name] = {"s": truthTable_s_list[i], "c": truthTable_c_list[i], "energy": energy_consumption_list[i], "steps": truthTable_steps_list[i]}

# Populate the dictionary for quality metrics

for j, kernel in enumerate(kernelname_list):
    results_dict[kernel] = {}
    for i, name in enumerate(nameApprox_list):
        results_dict[kernel][name] = {"ssi": list(), "psnr": list(), "energy_con": list(), "steps": list()}


def Adder(a, b, c, approxAlgo = "exact Semi Parallel [10]"):
    '''when no Algo is given use the base our approx is based on'''
    global loaded_dict
    if a==0 and b==0 and c==0:
        s = loaded_dict[approxAlgo]["s"][0]
        c_out = loaded_dict[approxAlgo]["c"][0]
        energy_consumption = loaded_dict[approxAlgo]["energy"][0]
        step = loaded_dict[approxAlgo]["steps"]
    elif a==0 and b==0 and c==1:
        s = loaded_dict[approxAlgo]["s"][1]
        c_out = loaded_dict[approxAlgo]["c"][1]
        energy_consumption = loaded_dict[approxAlgo]["energy"][1]
        step = loaded_dict[approxAlgo]["steps"]
    elif a==0 and b==1 and c==0:
        s = loaded_dict[approxAlgo]["s"][2]
        c_out = loaded_dict[approxAlgo]["c"][2]
        energy_consumption = loaded_dict[approxAlgo]["energy"][2]
        step = loaded_dict[approxAlgo]["steps"]
    elif a==0 and b==1 and c==1:
        s = loaded_dict[approxAlgo]["s"][3]
        c_out = loaded_dict[approxAlgo]["c"][3]
        energy_consumption = loaded_dict[approxAlgo]["energy"][3]
        step = loaded_dict[approxAlgo]["steps"]
    elif a==1 and b==0 and c==0:
        s = loaded_dict[approxAlgo]["s"][4]
        c_out = loaded_dict[approxAlgo]["c"][4]
        energy_consumption = loaded_dict[approxAlgo]["energy"][4]
        step = loaded_dict[approxAlgo]["steps"]
    elif a==1 and b==0 and c==1:
        s = loaded_dict[approxAlgo]["s"][5]
        c_out = loaded_dict[approxAlgo]["c"][5]
        energy_consumption = loaded_dict[approxAlgo]["energy"][5]
        step = loaded_dict[approxAlgo]["steps"]
    elif a==1 and b==1 and c==0:
        s = loaded_dict[approxAlgo]["s"][6]
        c_out = loaded_dict[approxAlgo]["c"][6]
        energy_consumption = loaded_dict[approxAlgo]["energy"][6]
        step = loaded_dict[approxAlgo]["steps"]
    elif a==1 and b==1 and c==1:
        s = loaded_dict[approxAlgo]["s"][7]
        c_out = loaded_dict[approxAlgo]["c"][7]
        energy_consumption = loaded_dict[approxAlgo]["energy"][7]
        step = loaded_dict[approxAlgo]["steps"]
    return s, c_out, energy_consumption, step

def My_Multiplier(a,b, approxAlgo, approxBit, blurrFlag):
    '''multiply two numbers reduce compuational effort for multiplication time 1 & -1'''
    energy = 0
    res = 0
    steps = 0

    if (a < -1 or a > 1) and (b < -1 or b > 1):

        a_wo_sign = abs(a)
        b_wo_sign = abs(b)

        if a_wo_sign > b_wo_sign:
            multcount = b
            multiplier = a
        else: 
            multcount = a
            multiplier = b

        if multcount < 0:
            multcount = multcount * (-1)
            multiplier = multiplier * (-1)
        
        for i in range(0, multcount):
            res, e, step  = MyNbitAdder(a=res, b=multiplier, Algo=approxAlgo, approx_until=approxBit)
            energy += e
            steps += step
    else:
        res = a * b
    if blurrFlag == True:
        res = res >> 4
    return res, energy, steps

def My_Mult(a, b, approxAlgo, approxBit, blurrFlag):
    '''go throw every pixel in 3x3 matrix'''
    energy = 0
    steps = 0
    res = np.zeros((a.shape[0],a.shape[1]))
    for k in range(a.shape[0]):
        for l in range(a.shape[1]):
            res[k,l], e, step = My_Multiplier(a=int(a[k,l]), b=int(b[k,l]), approxAlgo=approxAlgo, approxBit=approxBit, blurrFlag=blurrFlag)
            energy += e
            steps += step
    return res, energy, steps


def MyconvLUT(image, kernel, approxAlgo, approxBit, blurrFlag):
    '''do the convolution on image and kernel '''
    image = np.array(image).astype(int)
    x,y = image.shape
    k,l = kernel.shape
    # print(x,y)
    resmatrix = 0

    image = np.pad(image, ((k // 2, k // 2),(l // 2, l // 2)), mode = "constant")
    image_shape = np.shape(image)
    kernel_shape = np.shape(kernel)

    res_shape1 = np.abs(image_shape[0] - kernel_shape[0]) + 1
    res_shape2 = np.abs(image_shape[1] - kernel_shape[1]) + 1

    res = np.zeros((res_shape1, res_shape2))
    energy = 0
    steps = 0
    for i in range(y):
        for j in range(x):
            resmatrix, ee, step1 = My_Mult(a=np.flip(kernel), b=image[i:i + kernel_shape[0], j:j + kernel_shape[1]], approxAlgo=approxAlgo, approxBit=approxBit, blurrFlag=blurrFlag)
            res[i, j], e, step2 = MySum(matrix=resmatrix, approxAlgo=approxAlgo, approxBit=approxBit)
            energy = energy + ee + e
            steps = steps + step1 + step2
    return res, energy, steps

def MySum(matrix, approxAlgo, approxBit):
    '''sumreduction'''
    res = 0
    energy = 0
    steps = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            res, e, step = MyNbitAdder(a=res, b=int(matrix[i,j]), Algo=approxAlgo, approx_until=approxBit)
            energy += e
            steps += step

    return res, energy, steps

def MyNbitAdder(a, b, Algo, approx_until):
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
        steps = 0
        
        sum_list = []
        #we want to do a bitwise addition
        for index, (bit1, bit2) in enumerate(zip(rev_a, rev_b) ):
            if index <= approx_until or 'exact' in Algo:
                #use approx_adder
                sum_element, carry_over, energy, step = Adder(a=int(bit1), b=int(bit2), c=int(carry_over), approxAlgo=Algo) 
            else:
                #use exact_adder
                sum_element, carry_over, energy, step = Adder(a=int(bit1), b=int(bit2), c=int(carry_over))    
            
            sum_list.append(int(sum_element))
            total_energy += energy
            steps += step
        
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
        return total_sum, total_energy, steps #total energy in pJ!
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


def checkFilePresent(name):
    if os.path.isfile(f'{name}.png'):
        return True
    else: 
        return False

def main(path):
    rows = 8
    bit_list = range(0,rows)

    algo_list = ["own Aprox [11]","C51 paper [13]","exact Serial [1]","Serial Aprox [2]", "SIAFA 1 [3]","SIAFA 2 [4]","SIAFA 3 [5]","SIAFA 4 [6]","exact Semi Serial [7]","Serial Aprox [8]", "exact parallel [9]","exact Semi Parallel [10]"]
    # algo_list = ["own Aprox [11]"]

    calcAllNewFlag = True 

    cam_img = io.imread(f"{path}resources/cameraman.jpg", )

    R_1 = cam_img[:, :, 0] 
    G_1 = cam_img[:, :, 1]
    B_1 = cam_img[:, :, 2]

    #formula for converting colour(RGB) to Gray Image scale Image
    Y_cam = (0.299 * np.array(R_1)) + (0.587 * np.array(G_1)) + (0.114 * np.array(B_1)) 
    Y_cam_int = Y_cam.astype(int)

    plt.imshow(Y_cam_int , cmap = "gray")

    # assign the correct values for carry sum and energy acccording to choosen Algorithm
        
    for kernel, kernel_name in zip(kernel_list, kernelname_list):
        
        blurrFlag = False
        exactconv = signal.convolve2d(Y_cam_int, kernel, mode = "same")
        
        np.save(f'{path}data_{kernel_name}/exact.npy', exactconv)
        
        plt.imsave(f'{path}data_{kernel_name}/exact.png', exactconv, cmap='gray')
        # loop throw all Bitpositions 
        for approxAlgo in algo_list:
            # loop throw all Algorithm
            for approxBit in bit_list:
                approx_pic = 0

                if approxBit >= 1 and 'exact' in approxAlgo:
                    continue

                print(f'kernel: {kernel_name} Algo: {approxAlgo} Bit: {approxBit}')
                if not calcAllNewFlag:
                    if checkFilePresent(f'{path}data_{kernel_name}/outputimage_{approxAlgo}_{approxBit}'):
                        continue
                

                approx_pic, total_energy, total_steps = MyconvLUT(image=Y_cam_int, 
                                                    kernel=kernel, 
                                                    approxAlgo=approxAlgo, 
                                                    approxBit=approxBit, 
                                                    blurrFlag=blurrFlag)
                try:
                    data_range = approx_pic.max() - approx_pic.min()
                    print(f'sum: {np.sum(exactconv-approx_pic)} mean: {np.mean(exactconv-approx_pic)}')
                    results_dict[kernel_name][approxAlgo]["ssi"].append(ssim(exactconv, approx_pic, data_range=data_range))
                    results_dict[kernel_name][approxAlgo]["psnr"].append(psnr(exactconv, approx_pic, data_range=data_range))
                    results_dict[kernel_name][approxAlgo]["energy_con"].append(total_energy)
                    results_dict[kernel_name][approxAlgo]["steps"].append(total_steps)
                except Exception as e:
                    print(f'error {e}')
                else:
                    print(f'psnr: {results_dict[kernel_name][approxAlgo]["psnr"][approxBit]} ssim: {results_dict[kernel_name][approxAlgo]["ssi"][approxBit]}')

                np.save(f'{path}data_{kernel_name}/outputimage_{approxAlgo}_{approxBit}.npy', approx_pic)
                plt.imsave(f'{path}data_{kernel_name}/outputimage_{approxAlgo}_{approxBit}.png', approx_pic, cmap='gray')
                
                # Write the data to the JSON file
                with open(f'{path}results.json', 'w') as json_file:
                    json.dump(results_dict, json_file, indent=4)

def test(kernelname):
    testimage = np.array([
        [0,0,0,0,0,0,0,0,0],
        [0,255,255,255,0,0,0,0,0],
        [0,255,255,255,0,0,0,0,0],
        [0,255,255,255,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0]
        ])

    # testimage = np.zeros(shape=(9,9))

    # for a in range(0,9):
    #     for b in range(0,9):
    #         testimage[a,b] = (a+1)*(b+1)

    print(testimage.shape)

    algo_list = ["own Aprox [11]"]
    

    for algo in algo_list:
        for bit in range(0,8):

            if kernelname == 'blurring':
                exact = signal.convolve2d(testimage, blurrKernel, mode = "same")
                # exact,_ = MyconvLUT(kernel=blurrKernel, image=testimage,approxBit=bit, blurrFlag=False, approxAlgo="exact Serial [1]")
            else:
                exact = signal.convolve2d(testimage, edgeDetectionKernel, mode = "same")
                # exact,_ = MyconvLUT(kernel=edgeDetectionKernel, image=testimage,approxBit=bit, blurrFlag=False, approxAlgo="exact Serial [1]")

            exact = exact.astype(int)
            # print(exact)
            # print('\n')

            if kernelname == 'blurring':
                exactconv,_,_ = MyconvLUT(kernel=blurrKernel, image=testimage,approxBit=bit, blurrFlag=False, approxAlgo=algo)
            else:
                exactconv,_,_ = MyconvLUT(kernel=edgeDetectionKernel, image=testimage,approxBit=bit, blurrFlag=False, approxAlgo=algo)

            exactconv = exactconv.astype(int)
            # print(exactconv)
            data_range = exactconv.max() - exactconv.min()
            # datarange = exactconv.max() - exactconv.min()
            print(f'kernel: {kernelname} algo: {algo} Bit: {bit} psnr: {round(psnr(exact, exactconv, data_range=data_range),2)} ssim: {round(ssim(exact, exactconv, data_range=data_range),2)}')


if __name__ == "__main__":
    path = ''
    # path = 'final_project/image_processing/'

    main(path)

    # test('blurring')
    # test('edge Detection')
