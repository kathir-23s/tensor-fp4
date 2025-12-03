#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <bitset>
#include <string>
#include "dtype/Types.h"

// Test FP4 (E2M1) -> FP32 Conversion
// Exhaustively checks all 16 possible 4-bit values.

struct TestCase {
    uint8_t bits;
    float expected_val;
    bool is_nan;
    bool is_inf;
    std::string desc;
};

int main() {
    std::vector<TestCase> cases = {
        // Positive Numbers
        {0b0000, 0.0f, false, false, "Positive Zero"},
        {0b0001, 0.5f, false, false, "0.5"}, 
        {0b0010, 1.0f, false, false, "1.0"}, 
        {0b0011, 1.5f, false, false, "1.5"}, 
        {0b0100, 2.0f, false, false, "2.0"}, 
        {0b0101, 3.0f, false, false, "3.0"}, 
        {0b0110, 4.0f, false, false, "4.0"},
        {0b0111, 6.0f, false, false, "6.0"},

        // Negative Numbers
        {0b1000, -0.0f, false, false, "Negative Zero"},
        {0b1001, -0.5f, false, false, "-0.5"},
        {0b1010, -1.0f, false, false, "-1.0"},
        {0b1011, -1.5f, false, false, "-1.5"},
        {0b1100, -2.0f, false, false, "-2.0"},
        {0b1101, -3.0f, false, false, "-3.0"},
        {0b1110, -4.0f, false, false, "-4.0"},
        {0b1111, -6.0f, false, false, "-6.0"}
    };

    int passed = 0;
    int total = 0;

    std::cout << "========================================" << std::endl;
    std::cout << "TEST: FP4 -> FP32 Conversion" << std::endl;
    std::cout << "========================================" << std::endl;

    for (const auto& tc : cases) {
        total++;
        OwnTensor::float4_e2m1_t val(tc.bits);
        float f = static_cast<float>(val);

        bool ok = (f == tc.expected_val);

        std::cout << "Bits: " << std::bitset<4>(tc.bits) 
                  << " | Expected: " << std::setw(10) << (tc.is_nan ? "NaN" : std::to_string(tc.expected_val))
                  << " | Got: " << std::setw(10) << f
                  << " | " << tc.desc 
                  << " -> " << (ok ? "[PASS]" : "[FAIL]") << std::endl;

        if (ok) passed++;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    return (passed == total) ? 0 : 1;
}
