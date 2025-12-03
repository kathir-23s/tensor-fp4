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
        {-1.0f, 0b1010, "-1.0 (Exact)"},

        // 2. Rounding (Midpoints & In-Between)
        // E2M1 Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0
        
        // 0.25 is midpoint between 0 and 0.5. Rounding? 
        // Implementation uses simple nearest. 0.25 -> 0.5 (if >= 0.25) or 0?
        // Code: if (abs_f < 0.25f) bits = 0; else ...
        // So < 0.25 is 0, >= 0.25 starts next bin.
        {0.1f, 0b0000, "0.1 -> 0 (Underflow/Round down)"},
        {0.24f, 0b0000, "0.24 -> 0"},
        {0.25f, 0b0001, "0.25 -> 0.5 (Round up)"}, 
        
        {0.74f, 0b0001, "0.74 -> 0.5"},
        {0.75f, 0b0010, "0.75 -> 1.0 (Round up)"},

        {1.24f, 0b0010, "1.24 -> 1.0"},
        {1.25f, 0b0011, "1.25 -> 1.5 (Round up)"},

        {1.74f, 0b0011, "1.74 -> 1.5"},
        {1.75f, 0b0100, "1.75 -> 2.0 (Round up)"},

        {2.49f, 0b0100, "2.49 -> 2.0"},
        {2.5f, 0b0101, "2.5 -> 3.0 (Round up)"},

        // 3. Out of Range (Overflow)
        // Max representable is now 6.0.
        {3.1f, 0b0101, "3.1 -> 3.0"},
        {3.49f, 0b0101, "3.49 -> 3.0"},
        {3.5f, 0b0110, "3.5 -> 4.0 (Round up)"},
        {4.0f, 0b0110, "4.0 -> 4.0 (Exact)"},
        {4.9f, 0b0110, "4.9 -> 4.0"},
        {5.0f, 0b0111, "5.0 -> 6.0 (Round up)"},
        {6.0f, 0b0111, "6.0 -> 6.0 (Exact)"},
        {100.0f, 0b0111, "100.0 -> 6.0 (Saturate)"},
        {-4.0f, 0b1110, "-4.0 -> -4.0"},
        {-6.0f, 0b1111, "-6.0 -> -6.0"},
        {-100.0f, 0b1111, "-100.0 -> -6.0 (Saturate)"},

        // 4. Special Values
        {std::numeric_limits<float>::infinity(), 0b0111, "Inf -> 6.0 (Saturate)"},
        {-std::numeric_limits<float>::infinity(), 0b1111, "-Inf -> -6.0 (Saturate)"},
        {std::numeric_limits<float>::quiet_NaN(), 0b0000, "NaN -> 0.0 (Safe default)"}
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

        // For NaN, we now expect 0 (0000)
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
