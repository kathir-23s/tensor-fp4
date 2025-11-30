# Summary and Insights

## What We Did
We successfully implemented two new data types for the Tensor library:
1.  **`float4_e2m1_t`**: A 4-bit floating point type stored in a `uint8_t`. It follows the E2M1 format (1 Sign, 2 Exponent, 1 Mantissa) with a bias of 1.
2.  **`float4_e2m1_2x_t`**: A packed container storing two `float4_e2m1_t` values in a single `uint8_t`.

These types were fully integrated into the library's type system:
-   Added to the `Dtype` enum.
-   Traits registered in `DtypeTraits.h` (allowing the library to query properties like size and name).
-   Promotion rules established (operations promote to `Float32`, similar to how `Float16` is handled on CPU).
-   Conversion logic implemented in `Types.h` and `DtypeCastUtils.h`.

## Insights from Tests
-   **E2M1 Format**: The format behaves as expected.
    -   Smallest positive normal: 0.5 (represented as `0 00 1` -> $0.M \times 2^{1-1} = 0.5 \times 1 = 0.5$). *Correction: Subnormal logic used.*
    -   Max finite value: 3.0.
    -   Inf and NaN are correctly handled (Exponents of all 1s).
-   **Rounding**: We implemented nearest rounding. Values like 2.9 correctly round to 3.0, and 4.0 overflows to Infinity.
-   **Packing/Unpacking**: The packed type `float4_e2m1_2x_t` works correctly, but we encountered a subtle C++ issue during implementation. Accessors like `get_low()` initially failed because bitwise operations promoted `uint8_t` to `int`, causing the `float` constructor to be called instead of the raw bits constructor. We fixed this by explicitly casting back to `uint8_t`.

## Integration Status
**Fully Integrated.**
The types are now first-class citizens in the `OwnTensor` library.
-   They can be used to create Tensors (e.g., `Tensor(shape, Dtype::Float4_e2m1)`).
-   They support basic arithmetic operations via automatic promotion to `Float32` (as defined in `DtypeTraits.h` and `DtypeCastUtils.h`).
-   They are ready for use in memory-constrained scenarios where 4-bit precision is sufficient.
