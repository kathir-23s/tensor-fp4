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

namespace detail_fp4 {

// Helper to convert float to fp4 (E2M1 - Modified)
// Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
// Sign bit + 3 bits magnitude
__device__ __host__ inline uint8_t float_to_fp4_e2m1(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 31) & 0x1;

    float abs_f = std::abs(f);

    // Clamping: if > 6, clamp to 6 (max value)
    // Also handles Inf/NaN by treating them as large values -> max value
    if (std::isnan(abs_f) || abs_f > 6.0f) {
        return (sign << 3) | 0x7; // 6
    }

    // Nearest rounding to: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    // Midpoints:
    // 0   <-> 0.5 : 0.25
    // 0.5 <-> 1   : 0.75
    // 1   <-> 1.5 : 1.25
    // 1.5 <-> 2   : 1.75
    // 2   <-> 3   : 2.5
    // 3   <-> 4   : 3.5
    // 4   <-> 6   : 5.0

    uint8_t bits = 0;
    if (abs_f < 0.25f) bits = 0;      // 0
    else if (abs_f < 0.75f) bits = 1; // 0.5
    else if (abs_f < 1.25f) bits = 2; // 1
    else if (abs_f < 1.75f) bits = 3; // 1.5
    else if (abs_f < 2.25f) bits = 4; // 2 (Note: 2.25 is midpoint between 2 and 2.5? No, next is 3. Midpoint 2.5)
                                      // Wait, 2->3 midpoint is 2.5.
                                      // Let's recheck thresholds.
                                      // 0, 0.5, 1, 1.5, 2, 3, 4, 6
    else if (abs_f < 2.5f) bits = 4;  // 2
    else if (abs_f < 3.5f) bits = 5;  // 3
    else if (abs_f < 5.0f) bits = 6;  // 4
    else bits = 7;                    // 6

    return (sign << 3) | bits;
}

__device__ __host__ inline float fp4_e2m1_to_float(uint8_t val) {
    uint8_t sign = (val >> 3) & 0x1;
    uint8_t mag = val & 0x7;

    float res = 0.0f;
    switch (mag) {
        case 0: res = 0.0f; break;
        case 1: res = 0.5f; break;
        case 2: res = 1.0f; break;
        case 3: res = 1.5f; break;
        case 4: res = 2.0f; break;
        case 5: res = 3.0f; break;
        case 6: res = 4.0f; break;
        case 7: res = 6.0f; break;
    }

    return sign ? -res : res;
}

} // namespace detail_fp4

/**
 * @brief FP4 (E2M1)
 * Format: Sign(1) + Exponent(2) + Mantissa(1)
 * Storage: uint8_t (lower 4 bits)
 */
struct float4_e2m1_t {
    uint8_t raw_bits; // Only lower 4 bits used

    __device__ __host__ float4_e2m1_t() : raw_bits(0) {}
    __device__ __host__ explicit float4_e2m1_t(float val) { raw_bits = detail_fp4::float_to_fp4_e2m1(val); }
    __device__ __host__ explicit float4_e2m1_t(uint8_t bits) : raw_bits(bits & 0xF) {} // Direct construction
    __device__ __host__ float4_e2m1_t(const float4_e2m1_t& other) : raw_bits(other.raw_bits) {}

    template <typename U, typename = ::std::enable_if_t<
        ::std::is_arithmetic_v<U> && !::std::is_same_v<std::decay_t<U>, float>
    >>
    __device__ __host__ explicit float4_e2m1_t(U val) {
        raw_bits = detail_fp4::float_to_fp4_e2m1(static_cast<float>(val));
    }

    __device__ __host__ operator float() const { return detail_fp4::fp4_e2m1_to_float(raw_bits); }

    // Assignment
    __device__ __host__ float4_e2m1_t& operator=(float val) {
        raw_bits = detail_fp4::float_to_fp4_e2m1(val);
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

    // Compound Assignment
    __device__ __host__ float4_e2m1_t& operator+=(const float4_e2m1_t& other) { *this = *this + other; return *this; }
    __device__ __host__ float4_e2m1_t& operator-=(const float4_e2m1_t& other) { *this = *this - other; return *this; }
    __device__ __host__ float4_e2m1_t& operator*=(const float4_e2m1_t& other) { *this = *this * other; return *this; }
    __device__ __host__ float4_e2m1_t& operator/=(const float4_e2m1_t& other) { *this = *this / other; return *this; }

    // Unary operators
    __device__ __host__ float4_e2m1_t operator-() const {
        return float4_e2m1_t(-(float)*this);
    }
};

/**
 * @brief FP4 Packed (2x E2M1 in uint8)
 * Format: High 4 bits = Value 1, Low 4 bits = Value 0
 */
struct float4_e2m1_2x_t {
    uint8_t raw_bits;

    __device__ __host__ float4_e2m1_2x_t() : raw_bits(0) {}
    __device__ __host__ explicit float4_e2m1_2x_t(uint8_t bits) : raw_bits(bits) {}

    __device__ __host__ explicit float4_e2m1_2x_t(float val) {
        float4_e2m1_t v(val);
        raw_bits = (v.raw_bits & 0xF) | ((v.raw_bits & 0xF) << 4);
    }
    
    // Construct from two fp4 values
    __device__ __host__ float4_e2m1_2x_t(float4_e2m1_t v0, float4_e2m1_t v1) {
        raw_bits = (v0.raw_bits & 0xF) | ((v1.raw_bits & 0xF) << 4);
    }

    // Accessors
    __device__ __host__ float4_e2m1_t get_low() const { return float4_e2m1_t(static_cast<uint8_t>(raw_bits & 0xF)); }
    __device__ __host__ float4_e2m1_t get_high() const { return float4_e2m1_t(static_cast<uint8_t>((raw_bits >> 4) & 0xF)); }
    
    __device__ __host__ void set_low(float4_e2m1_t v) { raw_bits = (raw_bits & 0xF0) | (v.raw_bits & 0xF); }
    __device__ __host__ void set_high(float4_e2m1_t v) { raw_bits = (raw_bits & 0x0F) | ((v.raw_bits & 0xF) << 4); }

    // Assignment
    __device__ __host__ float4_e2m1_2x_t& operator=(float val) {
        float4_e2m1_t v(val);
        set_low(v);
        set_high(v);
        return *this;
    }

    // Conversion
    __device__ __host__ operator float() const { return (float)get_low(); }
    __device__ __host__ operator double() const { return (double)get_low(); }
    __device__ __host__ explicit operator float4_e2m1_t() const { return get_low(); }
    __device__ __host__ explicit operator bool() const { return raw_bits != 0; }
    __device__ __host__ explicit operator uint8_t() const { return raw_bits; }
    __device__ __host__ explicit operator int8_t() const { return (int8_t)raw_bits; }
    __device__ __host__ explicit operator uint16_t() const { return (uint16_t)raw_bits; }
    __device__ __host__ explicit operator int16_t() const { return (int16_t)raw_bits; }
    __device__ __host__ explicit operator uint32_t() const { return (uint32_t)raw_bits; }
    __device__ __host__ explicit operator int32_t() const { return (int32_t)raw_bits; }
    __device__ __host__ explicit operator uint64_t() const { return (uint64_t)raw_bits; }
    __device__ __host__ explicit operator int64_t() const { return (int64_t)raw_bits; }

    // Unary operators
    __device__ __host__ float4_e2m1_2x_t operator-() const {
        return float4_e2m1_2x_t(-(float)*this);
    }

    // Comparison
    __device__ __host__ bool operator==(const float4_e2m1_2x_t& other) const { return raw_bits == other.raw_bits; }
    __device__ __host__ bool operator!=(const float4_e2m1_2x_t& other) const { return raw_bits != other.raw_bits; }
    __device__ __host__ bool operator<(const float4_e2m1_2x_t& other) const { return raw_bits < other.raw_bits; }
    __device__ __host__ bool operator>(const float4_e2m1_2x_t& other) const { return raw_bits > other.raw_bits; }
    __device__ __host__ bool operator<=(const float4_e2m1_2x_t& other) const { return raw_bits <= other.raw_bits; }
    __device__ __host__ bool operator>=(const float4_e2m1_2x_t& other) const { return raw_bits >= other.raw_bits; }

    // Arithmetic
    __device__ __host__ float4_e2m1_2x_t operator+(const float4_e2m1_2x_t& other) const {
        return float4_e2m1_2x_t(get_low() + other.get_low(), get_high() + other.get_high());
    }
    __device__ __host__ float4_e2m1_2x_t operator-(const float4_e2m1_2x_t& other) const {
        return float4_e2m1_2x_t(get_low() - other.get_low(), get_high() - other.get_high());
    }
    __device__ __host__ float4_e2m1_2x_t operator*(const float4_e2m1_2x_t& other) const {
        return float4_e2m1_2x_t(get_low() * other.get_low(), get_high() * other.get_high());
    }
    __device__ __host__ float4_e2m1_2x_t operator/(const float4_e2m1_2x_t& other) const {
        return float4_e2m1_2x_t(get_low() / other.get_low(), get_high() / other.get_high());
    }

    // Compound Assignment
    __device__ __host__ float4_e2m1_2x_t& operator+=(const float4_e2m1_2x_t& other) { *this = *this + other; return *this; }
    __device__ __host__ float4_e2m1_2x_t& operator-=(const float4_e2m1_2x_t& other) { *this = *this - other; return *this; }
    __device__ __host__ float4_e2m1_2x_t& operator*=(const float4_e2m1_2x_t& other) { *this = *this * other; return *this; }
    __device__ __host__ float4_e2m1_2x_t& operator/=(const float4_e2m1_2x_t& other) { *this = *this / other; return *this; }
}; // Closing brace for float4_e2m1_2x_t
