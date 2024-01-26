import numpy as np

def custom_vectorized_add(a, b):
    # Your custom vectorized adder implementation
    result = a ^ b  # XOR operation for sum
    carry = (a & b) << 1  # AND operation for carry
    return result, carry

class CustomInt8Adder:
    def __init__(self, value):
        # Ensure value is a signed 8-bit integer
        if not (-128 <= value <= 127):
            raise ValueError("Value must be a signed 8-bit integer (-128 to 127).")

        self.value = np.int8(value)

    def __add__(self, other):
        if isinstance(other, CustomInt8Adder):
            # Perform custom addition using vectorized adder function
            result_value, carry = custom_vectorized_add(self.value, other.value)

            # Create a new instance with the result value
            result = CustomInt8Adder(int(result_value))

            # Set the carry bit in the result
            result.carry = int(carry[-1])  # Consider the last carry bit

            return result
        else:
            raise TypeError("Unsupported operand type. Must be an instance of CustomInt8Adder.")

# Example usage
value1 = CustomInt8Adder(100)
value2 = CustomInt8Adder(25)

result = value1 + value2
print("Sum:", result.value)
print("Carry:", result.carry)


