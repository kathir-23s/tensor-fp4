# FP4 Implementation Plan

## Goal
Implement FP4 (E2M1) and packed FP4 (2x) support for the CPU side of the Tensor library.

## Plan
1.  **Define Types**:
    -   Create `float4_e2m1_t` struct in `Types.h` using `uint8_t` storage (lower 4 bits).
    -   Create `float4_e2m1_2x_t` struct in `Types.h` for packed storage (2 values in 1 byte).
    -   Implement conversion logic (Float <-> FP4) following E2M1 format (1 Sign, 2 Exp, 1 Mantissa, Bias 1).

2.  **Integrate into Dtype System**:
    -   Add `Float4_e2m1` and `Float4_e2m1_2x` to `Dtype` enum in `Dtype.h`.
    -   Register traits (name, size, properties) in `DtypeTraits.h`.
    -   Update type promotion rules in `DtypeTraits.h` (promote to Float32 for ops).
    -   Update casting utilities in `DtypeCastUtils.h`.

3.  **Verification**:
    -   Create a dedicated test suite `Tests/fp4_tests/test_fp4.cpp`.
    -   Test bit-exact conversion for all 16 FP4 values.
    -   Test rounding behavior from Float32.
    -   Test packing and unpacking functionality.
