#include <iostream>
#include <cstdint>

int8_t add(int8_t a, int8_t b) {

    return a + b;
}

int main() {
    int8_t num1, num2;

    // Test 1
    int8_t result1 = add(2, 3);
    std::cout << "Test 1: 2 + 3 = " << static_cast<int>(result1) << " (Expected: 5)\n";

    // Test 2
    int8_t result2 = add(-1, 5);
    std::cout << "Test 2: -1 + 5 = " << static_cast<int>(result2) << " (Expected: 4)\n";

    // Test 3
    int8_t result3 = add(0, 0);
    std::cout << "Test 3: 0 + 0 = " << static_cast<int>(result3) << " (Expected: 0)\n";

    // Test 4
    int8_t result4 = add(-10, -5);
    std::cout << "Test 4: -10 + -5 = " << static_cast<int>(result4) << " (Expected: -15)\n";

    // User input
    std::cout << "Enter the first integer: ";
    int tempNum1;
    std::cin >> tempNum1;
    num1 = static_cast<int8_t>(tempNum1);

    std::cout << "Enter the second integer: ";
    int tempNum2;
    std::cin >> tempNum2;
    num2 = static_cast<int8_t>(tempNum2);

    // Check if the input is within int8_t range
    if (num1 < INT8_MIN || num1 > INT8_MAX || num2 < INT8_MIN || num2 > INT8_MAX) {
        std::cerr << "Error: Input is out of int8_t range.\n";
        return EXIT_FAILURE;
    }


    // Addition
    int8_t result = add(num1, num2);

    // Output
    std::cout << "The sum of " << static_cast<int>(num1) << " and " << static_cast<int>(num2) 
              << " is: " << static_cast<int>(result) << std::endl;

    return 0;
}
