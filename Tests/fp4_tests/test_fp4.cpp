#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <bitset>
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"
#include "dtype/DtypeCastUtils.h"

using namespace OwnTensor;

void test_fp4_conversion() {
    std::cout << "Testing FP4 E2M1 Conversion..." << std::endl;
    
    // Table of expected values based on E2M1 (Bias 1)
    // S E E M
    // 0 00 0 = 0
    // 0 00 1 = 0.5
    // 0 01 0 = 1.0
    // 0 01 1 = 1.5
    // 0 10 0 = 2.0
    // 0 10 1 = 3.0
    // 0 11 0 = Inf
    // 0 11 1 = NaN
    
    struct TestCase {
        uint8_t bits;
        float expected;
        bool is_nan;
        bool is_inf;
    };

    std::vector<TestCase> cases = {
        {0b0000, 0.0f, false, false},
        {0b0001, 0.5f, false, false},
        {0b0010, 1.0f, false, false},
        {0b0011, 1.5f, false, false},
        {0b0100, 2.0f, false, false},
        {0b0101, 3.0f, false, false},
        {0b0110, 0.0f, false, true}, // Inf
        {0b0111, 0.0f, true, false}, // NaN
        // Negatives
        {0b1000, -0.0f, false, false},
        {0b1001, -0.5f, false, false},
        {0b1010, -1.0f, false, false},
        {0b1011, -1.5f, false, false},
        {0b1100, -2.0f, false, false},
        {0b1101, -3.0f, false, false},
        {0b1110, 0.0f, false, true}, // -Inf
        {0b1111, 0.0f, true, false}  // NaN
    };

    for (const auto& tc : cases) {
        float4_e2m1_t val(tc.bits);
        float f = static_cast<float>(val);
        
        std::cout << "Bits: " << std::bitset<4>(tc.bits) << " -> Float: " << f;
        
        if (tc.is_nan) {
            if (std::isnan(f)) std::cout << " [PASS]" << std::endl;
            else std::cout << " [FAIL] Expected NaN" << std::endl;
        } else if (tc.is_inf) {
            if (std::isinf(f)) std::cout << " [PASS]" << std::endl;
            else std::cout << " [FAIL] Expected Inf" << std::endl;
        } else {
            if (std::abs(f - tc.expected) < 1e-5f) std::cout << " [PASS]" << std::endl;
            else std::cout << " [FAIL] Expected " << tc.expected << std::endl;
        }
    }
}

void test_float_to_fp4() {
    std::cout << "\nTesting Float -> FP4 Conversion..." << std::endl;
    
    struct TestCase {
        float input;
        uint8_t expected_bits;
    };
    
    std::vector<TestCase> cases = {
        {0.0f, 0b0000},
        {0.2f, 0b0000}, // Round down to 0
        {0.3f, 0b0001}, // Round up to 0.5 (midpoint 0.25)
        {0.5f, 0b0001},
        {0.75f, 0b0010}, // Round to 1.0 (midpoint 0.75, tie break up?) Logic says <0.75 is 0.5. >=0.75 is 1.0.
        {1.0f, 0b0010},
        {1.25f, 0b0011}, // Round to 1.5
        {1.5f, 0b0011},
        {2.0f, 0b0100},
        {2.9f, 0b0101}, // Round to 3.0
        {3.0f, 0b0101},
        {4.0f, 0b0110}, // Overflow to Inf
        {-1.0f, 0b1010},
        {-3.0f, 0b1101}
    };

    for (const auto& tc : cases) {
        float4_e2m1_t val(tc.input);
        uint8_t bits = val.raw_bits & 0xF;
        
        std::cout << "Input: " << tc.input << " -> Bits: " << std::bitset<4>(bits);
        
        if (bits == tc.expected_bits) std::cout << " [PASS]" << std::endl;
        else std::cout << " [FAIL] Expected " << std::bitset<4>(tc.expected_bits) << std::endl;
    }
}

void test_fp4_packing() {
    std::cout << "\nTesting FP4 Packing..." << std::endl;
    
    float4_e2m1_t v1(1.0f); // 0010
    float4_e2m1_t v2(2.0f); // 0100
    
    float4_e2m1_2x_t packed(v1, v2);
    
    // Expect: v2 (high) | v1 (low) -> 0100 0010
    uint8_t expected = 0b01000010;
    
    std::cout << "Packed: " << std::bitset<8>(packed.raw_bits);
    if (packed.raw_bits == expected) std::cout << " [PASS]" << std::endl;
    else std::cout << " [FAIL] Expected " << std::bitset<8>(expected) << std::endl;
    
    float4_e2m1_t u1 = packed.get_low();
    float4_e2m1_t u2 = packed.get_high();
    
    if (static_cast<float>(u1) == 1.0f && static_cast<float>(u2) == 2.0f) {
        std::cout << "Unpacking [PASS]" << std::endl;
    } else {
        std::cout << "Unpacking [FAIL] Got: " << static_cast<float>(u1) << ", " << static_cast<float>(u2) << std::endl;
    }
}

int main() {
    test_fp4_conversion();
    test_float_to_fp4();
    test_fp4_packing();
    return 0;
}
