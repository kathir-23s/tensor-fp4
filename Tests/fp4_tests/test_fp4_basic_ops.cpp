#include <iostream>
#include <vector>
#include <cmath>
#include "core/Tensor.h"
#include "ops/TensorOps.h"
#include "dtype/Types.h"

using namespace OwnTensor;

// Helper to check results
bool check_tensor(const Tensor& t, const std::vector<float>& expected) {
    if (t.numel() != expected.size()) return false;
    Tensor t_f32 = t.as_type(Dtype::Float32);
    const float* data = t_f32.data<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        if (std::abs(data[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

int main() {
    std::cout << "Running FP4 Basic Ops Test..." << std::endl;
    
    // 1. Setup Data
    // A = [1.0, 2.0]
    // B = [0.5, 3.0]
    Tensor A(Shape{{2}}, Dtype::Float4_e2m1);
    Tensor B(Shape{{2}}, Dtype::Float4_e2m1);
    
    std::vector<float4_e2m1_t> data_a = {float4_e2m1_t(1.0f), float4_e2m1_t(2.0f)};
    std::vector<float4_e2m1_t> data_b = {float4_e2m1_t(0.5f), float4_e2m1_t(3.0f)};
    
    A.set_data(data_a);
    B.set_data(data_b);

    // 2. Addition
    // [1.0+0.5, 2.0+3.0] = [1.5, 5.0->6.0(saturate)]
    // Note: 2.0+3.0 = 5.0. Max FP4 is 6.0. 5.0 rounds to 6.0.
    Tensor C_add = A + B;
    if (check_tensor(C_add, {1.5f, 6.0f})) {
        std::cout << "Addition: PASS" << std::endl;
    } else {
        std::cout << "Addition: FAIL" << std::endl;
        C_add.display();
    }

    // 3. Subtraction
    // [1.0-0.5, 2.0-3.0] = [0.5, -1.0]
    Tensor C_sub = A - B;
    if (check_tensor(C_sub, {0.5f, -1.0f})) {
        std::cout << "Subtraction: PASS" << std::endl;
    } else {
        std::cout << "Subtraction: FAIL" << std::endl;
        C_sub.display();
    }

    // 4. Multiplication
    // [1.0*0.5, 2.0*3.0] = [0.5, 6.0->6.0(saturate)]
    Tensor C_mul = A * B;
    if (check_tensor(C_mul, {0.5f, 6.0f})) {
        std::cout << "Multiplication: PASS" << std::endl;
    } else {
        std::cout << "Multiplication: FAIL" << std::endl;
        C_mul.display();
    }

    // 5. Division
    // [1.0/0.5, 2.0/3.0] = [2.0, 0.666...]
    // 0.666... rounds to 0.5 (0.5) or 1.0 (1.0)?
    // 0.666 is closer to 0.5 (diff 0.166) than 1.0 (diff 0.333). -> 0.5.
    Tensor C_div = A / B;
    if (check_tensor(C_div, {2.0f, 0.5f})) {
        std::cout << "Division: PASS" << std::endl;
    } else {
        std::cout << "Division: FAIL" << std::endl;
    }

    return 0;
}
