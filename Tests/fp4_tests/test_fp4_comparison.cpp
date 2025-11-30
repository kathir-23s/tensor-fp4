#include <iostream>
#include <vector>
#include "core/Tensor.h"
#include "ops/TensorOps.h"
#include "dtype/Types.h"

using namespace OwnTensor;

int main() {
    std::cout << "Running FP4 Comparison Test..." << std::endl;

    // A: [1.0, 2.0, 3.0]
    Tensor A(Shape{{3}}, Dtype::Float4_e2m1);
    A.set_data<float4_e2m1_t>({float4_e2m1_t(1.0f), float4_e2m1_t(2.0f), float4_e2m1_t(3.0f)});

    // B: [1.0, 3.0, 2.0]
    Tensor B(Shape{{3}}, Dtype::Float4_e2m1);
    B.set_data<float4_e2m1_t>({float4_e2m1_t(1.0f), float4_e2m1_t(3.0f), float4_e2m1_t(2.0f)});

    // 1. Equal (==) -> [T, F, F]
    Tensor eq = (A == B);
    const bool* d_eq = eq.data<bool>();
    if (d_eq[0] && !d_eq[1] && !d_eq[2]) std::cout << "Equal: PASS" << std::endl;
    else std::cout << "Equal: FAIL" << std::endl;

    // 2. Less Than (<) -> [F, T, F]
    Tensor lt = (A < B);
    const bool* d_lt = lt.data<bool>();
    if (!d_lt[0] && d_lt[1] && !d_lt[2]) std::cout << "Less Than: PASS" << std::endl;
    else std::cout << "Less Than: FAIL" << std::endl;

    // 3. Greater Than (>) -> [F, F, T]
    Tensor gt = (A > B);
    const bool* d_gt = gt.data<bool>();
    if (!d_gt[0] && !d_gt[1] && d_gt[2]) std::cout << "Greater Than: PASS" << std::endl;
    else std::cout << "Greater Than: FAIL" << std::endl;

    return 0;
}
