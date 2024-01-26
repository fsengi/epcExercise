import numpy as np

def custom_adder_bit(A, B, Cin, sum_table, carry_table):
    index = A + 2 * B + 4 * Cin
    return bool(sum_table[index]), bool(carry_table[index])

def custom_adder_int8(A, B):
    assert A.dtype == B.dtype == np.int8, "Input vectors must be of dtype int8"

    S = np.zeros(8, dtype=bool)
    C = np.zeros(8, dtype=bool)

    for i in range(8):
        S[i], C[i] = custom_adder_bit((A >> i) & 1, (B >> i) & 1, C[i - 1] if i > 0 else 0, sum_table, carry_table)

    return np.packbits(S), np.packbits(C)

# Define truth tables for the sum bit and the carry bit using boolean values
sum_table = np.array([False, True, True, False, True, False, False, True], dtype=bool)
carry_table = np.array([False, False, False, True, False, True, True, True], dtype=bool)

# Test the custom_adder_int8 function
A = np.int8(34)  # Binary: 10101010
B = np.int8(85)   # Binary: 01010101
S, Cout = custom_adder_int8(A, B)
print(f"Custom Adder Int8 Test: A={int(A)}, B={int(B)} -> S={int(S)}, Cout={int(Cout)}")
