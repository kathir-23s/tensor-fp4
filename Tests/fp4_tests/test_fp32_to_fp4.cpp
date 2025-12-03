#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <bitset>
#include <string>
#include "dtype/Types.h"

// Test FP32 -> FP4 (E2M1) Conversion
// Checks exact matches, rounding behavior, and out-of-range handling.

struct TestCase {
    float input;
    uint8_t expected_bits;
    std::string desc;
};

int main() {
    std::vector<TestCase> cases = {
        // 1. Exact Matches
        {0.0f, 0b0000, "Zero"},
        {0.5f, 0b0001, "0.5 (Exact)"},
        {1.0f, 0b0010, "1.0 (Exact)"},
        {1.5f, 0b0011, "1.5 (Exact)"},
        {2.0f, 0b0100, "2.0 (Exact)"},
        {3.0f, 0b0101, "3.0 (Exact)"},
        {4.0f, 0b0110, "4.0 (Exact)"},
        {6.0f, 0b0111, "6.0 (Exact)"},
        {-1.0f, 0b1010, "-1.0 (Exact)"},

        // 2. Rounding (Midpoints & In-Between)
        // Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        
        {0.1f, 0b0000, "0.1 -> 0"},
        {0.24f, 0b0000, "0.24 -> 0"},
        {0.25f, 0b0001, "0.25 -> 0.5"}, 
        
        {0.74f, 0b0001, "0.74 -> 0.5"},
        {0.75f, 0b0010, "0.75 -> 1.0"},

        {1.24f, 0b0010, "1.24 -> 1.0"},
        {1.25f, 0b0011, "1.25 -> 1.5"},

        {1.74f, 0b0011, "1.74 -> 1.5"},
        {1.75f, 0b0100, "1.75 -> 2.0"}, // Midpoint 1.75? 
        // 1.5 <-> 2.0 midpoint is 1.75. Code: < 1.75 is 1.5. >= 1.75 is 2.0?
        // Let's check code: else if (abs_f < 1.75f) bits = 3; // 1.5
        // So 1.75 goes to next bin (bits=4 -> 2.0). Correct.

        {2.49f, 0b0100, "2.49 -> 2.0"},
        {2.5f, 0b0101, "2.5 -> 3.0"},

        {3.49f, 0b0101, "3.49 -> 3.0"},
        {3.5f, 0b0110, "3.5 -> 4.0"},

        {4.9f, 0b0110, "4.9 -> 4.0"},
        {5.0f, 0b0111, "5.0 -> 6.0"},

        // 3. Out of Range (Overflow)
        // Max representable is 6.0.
        {6.1f, 0b0111, "6.1 -> 6.0 (Clamped)"},
        {100.0f, 0b0111, "100.0 -> 6.0 (Clamped)"},
        {-7.0f, 0b1111, "-7.0 -> -6.0 (Clamped)"},
        {-100.0f, 0b1111, "-100.0 -> -6.0 (Clamped)"},

        // 4. Special Values
        {std::numeric_limits<float>::infinity(), 0b0111, "Inf -> 6.0"},
        {-std::numeric_limits<float>::infinity(), 0b1111, "-Inf -> -6.0"},
        {std::numeric_limits<float>::quiet_NaN(), 0b0111, "NaN -> 6.0"}
    };

    int passed = 0;
    int total = 0;

    std::cout << "========================================" << std::endl;
    std::cout << "TEST: FP32 -> FP4 Conversion" << std::endl;
    std::cout << "========================================" << std::endl;

    for (const auto& tc : cases) {
        total++;
        OwnTensor::float4_e2m1_t val(tc.input);
        uint8_t bits = val.raw_bits;

        // For NaN, exact bit match might vary, but we expect 0111 (7) or 1111 (15)
        // But with new logic, NaN -> 6.0 (0111).
        bool ok = (bits == tc.expected_bits);

        std::cout << "Input: " << std::setw(10) << tc.input 
                  << " | Expected: " << std::bitset<4>(tc.expected_bits)
                  << " | Got: " << std::bitset<4>(bits)
                  << " | " << tc.desc 
                  << " -> " << (ok ? "[PASS]" : "[FAIL]") << std::endl;

        if (ok) passed++;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    return (passed == total) ? 0 : 1;
}
