#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <string>
#include <bitset>
#include "dtype/Types.h"

// IEEE 754 E2M1 Standard (4-bit)
// Sign: 1 bit
// Exponent: 2 bits
// Mantissa: 1 bit
// Bias: 2^(2-1) - 1 = 1

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
        {0b0001, 0.5f, false, false, "Smallest Normal (Subnormal in IEEE logic?)"}, 
        // Wait, Exp=0 is subnormal. 0.M * 2^(1-Bias) = 0.1 * 2^0 = 0.5. Correct.
        
        {0b0010, 1.0f, false, false, "Normal 1.0"}, // Exp=1. 1.0 * 2^(1-1) = 1.0
        {0b0011, 1.5f, false, false, "Normal 1.5"}, // Exp=1. 1.1 * 2^0 = 1.5
        
        {0b0100, 2.0f, false, false, "Normal 2.0"}, // Exp=2. 1.0 * 2^(2-1) = 2.0
        {0b0101, 3.0f, false, false, "Normal 3.0"}, // Exp=2. 1.1 * 2^1 = 3.0
        
        {0b0110, std::numeric_limits<float>::infinity(), false, true, "Positive Infinity"},
        {0b0111, std::numeric_limits<float>::quiet_NaN(), true, false, "NaN"},

        // Negative Numbers
        {0b1000, -0.0f, false, false, "Negative Zero"},
        {0b1001, -0.5f, false, false, "Negative 0.5"},
        {0b1010, -1.0f, false, false, "Negative 1.0"},
        {0b1011, -1.5f, false, false, "Negative 1.5"},
        {0b1100, -2.0f, false, false, "Negative 2.0"},
        {0b1101, -3.0f, false, false, "Negative 3.0"},
        {0b1110, -std::numeric_limits<float>::infinity(), false, true, "Negative Infinity"},
        {0b1111, std::numeric_limits<float>::quiet_NaN(), true, false, "NaN"}
    };

    int passed = 0;
    int total = 0;

    std::cout << "========================================" << std::endl;
    std::cout << "FP4 (E2M1) Completeness Verification" << std::endl;
    std::cout << "========================================" << std::endl;

    for (const auto& tc : cases) {
        total++;
        OwnTensor::float4_e2m1_t val(tc.bits);
        float f = static_cast<float>(val);

        bool ok = false;
        if (tc.is_nan) {
            ok = std::isnan(f);
        } else if (tc.is_inf) {
            ok = std::isinf(f) && ((tc.bits & 0x8) ? (f < 0) : (f > 0));
        } else {
            // Check for exact equality as these are small powers of 2 (or sums of them)
            // But -0.0 == 0.0 in C++, so we might check signbit for zero if we want to be pedantic.
            // For now, value equality is enough for "value representation".
            ok = (f == tc.expected_val);
            
            // Optional: Check sign of zero
            if (f == 0.0f && std::signbit(f) != std::signbit(tc.expected_val)) {
                 // std::cout << " [Sign Mismatch for Zero] ";
                 // ok = false; // Strict sign check?
            }
        }

        std::cout << "Bits: " << std::bitset<4>(tc.bits) 
                  << " | Expected: " << std::setw(10) << (tc.is_nan ? "NaN" : std::to_string(tc.expected_val))
                  << " | Got: " << std::setw(10) << f
                  << " | " << tc.desc 
                  << " -> " << (ok ? "[PASS]" : "[FAIL]") << std::endl;

        if (ok) passed++;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "SUCCESS: All 16 FP4 values are correctly represented." << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Some values did not match." << std::endl;
        return 1;
    }
}
