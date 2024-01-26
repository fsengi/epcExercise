import numpy as np

# Convert uint8 array to boolean array
def tobool(uint8_array):
    bool_array = np.unpackbits(uint8_array.view(np.uint8))
    return bool_array.astype(bool)

# Convert boolean array to uint8 array
def toint(bool_array):
    uint8_array = np.packbits(bool_array.astype(np.uint8))
    return uint8_array.view(np.uint8)

# Example usage
original_uint8_array = np.array([255], dtype=np.uint8)

bool_array = tobool(original_uint8_array)
print("Boolean Array:", bool_array.astype(int))  # Convert boolean array to integers for printing

reversed_uint8_array = toint(bool_array)
print("Reversed uint8 Array:", reversed_uint8_array)




def binadd(input1, input2, input3):
    # Creating all possible combinations of boolean inputs
    input_combinations = np.array([
        [False, False, False],
        [False, False, True],
        [False, True, False],
        [False, True, True],
        [True, False, False],
        [True, False, True],
        [True, True, False],
        [True, True, True]
    ], dtype=bool)

    # Finding the index of the current input combination
    index = np.where(np.all(input_combinations == [input1, input2, input3], axis=1))[0][0]

    # Filling in the output combinations based on the provided information
    output_combinations = np.array([
        [True, False],
        [True, False],
        [True, False],
        [True, False],
        [True, False],
        [False, True],
        [False, True],
        [False, True]
    ], dtype=bool)

    # Retrieving the output combination based on the index
    output1, output2 = output_combinations[index]

    return output1, output2

# Example usage
# result = binadd(True, True, True)
# print(result)




def array_adder(ari, aro):
    res = np.zeros(8, dtype=bool)
    car = False
    for i in reversed(range(8)):
        # print(i)
        res[i], car = binadd(ari[i], aro[i], car)
    #     print(ari[i], aro[i], car)
    #     print(res[i], car)
    # print(res)
    return res


u = np.uint8(345)

print("look here", tobool(u))


print("result:" ,toint(array_adder(tobool(u), tobool(u))))

# print(np.uint8(True))

# print(toint(np.array([False, False,True, False,False, False,False, False])))
# print(toint(tobool(np.uint8(44))))