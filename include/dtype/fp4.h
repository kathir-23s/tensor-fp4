#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

// CUDA COMPATIBILITY MACROS (Redefined here or rely on Types.h being included first? 
// Better to be self-contained or use a common macro file. 
// For now, I will assume standard macros are available or re-define them safely if needed.
// Actually, Types.h has them. If I include fp4.h in Types.h, I should be careful.
// But usually these macros are project-wide. Let's just use them assuming they are defined if included in Types.h context, 
// OR define them if missing to be safe.)

#ifndef __CUDACC__
    #ifndef __device__
        #define __device__
    #endif
    #ifndef __host__
        #define __host__
    #endif
#endif

namespace detail {

// Helper to convert float to fp4 (E2M1)
// Format: S(1) E(2) M(1)
// Bias: 1
// Helper to convert float to fp4 (E2M1)
// Format: S(1) E(2) M(1)
// Bias: 1
// New Mapping (No Inf/NaN):
// 000: 0
// 001: 0.5 (Subnormal)
// 010: 1.0 (Norm, E=1, M=0)
// 011: 1.5 (Norm, E=1, M=1)
// 100: 2.0 (Norm, E=2, M=0)
// 101: 3.0 (Norm, E=2, M=1)
// 110: 4.0 (Norm, E=3, M=0) - Previously Inf
// 111: 6.0 (Norm, E=3, M=1) - Previously NaN
__device__ __host__ inline uint8_t float_to_fp4_e2m1(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 31) & 0x1;
    
    // Handle NaN -> 0 (Safe default)
    if (std::isnan(f)) {
        return (sign << 3) | 0x0; 
    }

    float abs_f = std::abs(f);

    // Hardcoded nearest rounding
    uint8_t bits = 0;
    if (abs_f < 0.25f) bits = 0;       // 0
    else if (abs_f < 0.75f) bits = 1;  // 0.5
    else if (abs_f < 1.25f) bits = 2;  // 1.0
    else if (abs_f < 1.75f) bits = 3;  // 1.5
    else if (abs_f < 2.5f) bits = 4;   // 2.0
    else if (abs_f < 3.5f) bits = 5;   // 3.0
    else if (abs_f < 5.0f) bits = 6;   // 4.0
    else bits = 7;                     // 6.0 (Saturate max)

    return (sign << 3) | bits;
}

__device__ __host__ inline float fp4_e2m1_to_float(uint8_t val) {
    uint8_t sign = (val >> 3) & 0x1;
    uint8_t exp = (val >> 1) & 0x3;
    uint8_t mant = val & 0x1;

    float res = 0.0f;
    if (exp == 0) {
        if (mant == 0) res = 0.0f;
        else res = 0.5f; // Subnormal: 0.1 * 2^0
    } else if (exp == 1) {
        // 1.M * 2^0
        res = (1.0f + mant * 0.5f); 
    } else if (exp == 2) {
        // 1.M * 2^1
        res = (1.0f + mant * 0.5f) * 2.0f;
    } else { // exp == 3
        // 1.M * 2^2 -> 4.0 or 6.0
        res = (1.0f + mant * 0.5f) * 4.0f;
    }

    return sign ? -res : res;
}

} // namespace detail

/**
 * @brief FP4 (E2M1)
 * Format: Sign(1) + Exponent(2) + Mantissa(1)
 * Storage: uint8_t (lower 4 bits)
 */
struct float4_e2m1_t {
    uint8_t raw_bits; // Only lower 4 bits used

    __device__ __host__ float4_e2m1_t() : raw_bits(0) {}
    __device__ __host__ explicit float4_e2m1_t(float val) { raw_bits = detail::float_to_fp4_e2m1(val); }
    __device__ __host__ explicit float4_e2m1_t(uint8_t bits) : raw_bits(bits & 0xF) {} // Direct construction
    __device__ __host__ float4_e2m1_t(const float4_e2m1_t& other) : raw_bits(other.raw_bits) {}

    template <typename U, typename = ::std::enable_if_t<
        ::std::is_arithmetic_v<U> && !::std::is_same_v<std::decay_t<U>, float>
    >>
    __device__ __host__ explicit float4_e2m1_t(U val) {
        raw_bits = detail::float_to_fp4_e2m1(static_cast<float>(val));
    }

    __device__ __host__ operator float() const { return detail::fp4_e2m1_to_float(raw_bits); }

    // Assignment
    __device__ __host__ float4_e2m1_t& operator=(float val) {
        raw_bits = detail::float_to_fp4_e2m1(val);
        return *this;
    }
    __device__ __host__ float4_e2m1_t& operator=(const float4_e2m1_t& other) {
        raw_bits = other.raw_bits;
        return *this;
    }

    // Comparison
    __device__ __host__ bool operator==(const float4_e2m1_t& other) const { return raw_bits == other.raw_bits; }
    __device__ __host__ bool operator!=(const float4_e2m1_t& other) const { return raw_bits != other.raw_bits; }
    
    // Basic Arithmetic (via float)
    __device__ __host__ float4_e2m1_t operator+(const float4_e2m1_t& other) const { return float4_e2m1_t(static_cast<float>(*this) + static_cast<float>(other)); }
    __device__ __host__ float4_e2m1_t operator-(const float4_e2m1_t& other) const { return float4_e2m1_t(static_cast<float>(*this) - static_cast<float>(other)); }
    __device__ __host__ float4_e2m1_t operator*(const float4_e2m1_t& other) const { return float4_e2m1_t(static_cast<float>(*this) * static_cast<float>(other)); }
    __device__ __host__ float4_e2m1_t operator/(const float4_e2m1_t& other) const { return float4_e2m1_t(static_cast<float>(*this) / static_cast<float>(other)); }
};

/**
 * @brief FP4 Packed (2x E2M1 in uint8)
 * Format: High 4 bits = Value 1, Low 4 bits = Value 0
 */
struct float4_e2m1_2x_t {
    uint8_t raw_bits;

    __device__ __host__ float4_e2m1_2x_t() : raw_bits(0) {}
    __device__ __host__ explicit float4_e2m1_2x_t(uint8_t bits) : raw_bits(bits) {}
    
    // Construct from two fp4 values
    __device__ __host__ float4_e2m1_2x_t(float4_e2m1_t v0, float4_e2m1_t v1) {
        raw_bits = (v0.raw_bits & 0xF) | ((v1.raw_bits & 0xF) << 4);
    }

    // Accessors
    __device__ __host__ float4_e2m1_t get_low() const { return float4_e2m1_t(static_cast<uint8_t>(raw_bits & 0xF)); }
    __device__ __host__ float4_e2m1_t get_high() const { return float4_e2m1_t(static_cast<uint8_t>((raw_bits >> 4) & 0xF)); }
    
    __device__ __host__ void set_low(float4_e2m1_t v) { raw_bits = (raw_bits & 0xF0) | (v.raw_bits & 0xF); }
    __device__ __host__ void set_high(float4_e2m1_t v) { raw_bits = (raw_bits & 0x0F) | ((v.raw_bits & 0xF) << 4); }
}; // Closing brace for float4_e2m1_2x_t
