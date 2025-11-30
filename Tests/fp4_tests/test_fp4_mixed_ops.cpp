#include <iostream>
#include <vector>
#include "core/Tensor.h"
#include "ops/TensorOps.h"
#include "dtype/Types.h"

using namespace OwnTensor;

int main() {
    std::cout << "Running FP4 Mixed Precision Test..." << std::endl;

    // A (FP4): [1.0, 2.0]
    Tensor A(Shape{{2}}, Dtype::Float4_e2m1);
    A.set_data<float4_e2m1_t>({float4_e2m1_t(1.0f), float4_e2m1_t(2.0f)});

    // B (FP32): [0.5, 0.1]
    Tensor B(Shape{{2}}, Dtype::Float32);
    B.set_data<float>({0.5f, 0.1f});

    // C = A + B
    // Should promote to FP32 (highest precision)
    // [1.0+0.5, 2.0+0.1] = [1.5, 2.1]
    Tensor C = A + B;

    if (C.dtype() != Dtype::Float32) {
        std::cout << "Promotion to FP32: FAIL (Got " << (int)C.dtype() << ")" << std::endl;
        return 1;
    }

    const float* data = C.data<float>();
    bool pass = true;
    if (data[0] != 1.5f) pass = false;
    if (std::abs(data[1] - 2.1f) > 1e-5f) pass = false;

    std::cout << "Mixed Add (FP4 + FP32): " << (pass ? "PASS" : "FAIL") << std::endl;

    return 0;
}
