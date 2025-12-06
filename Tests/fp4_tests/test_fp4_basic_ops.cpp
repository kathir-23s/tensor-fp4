#include <iostream>
#include <vector>
#include <cmath>
#include "TensorLib.h"

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

// Helper to print tensor metadata
void print_tensor_metadata(const std::string& name, const Tensor& t) {
    std::cout << "\n--- " << name << " Metadata ---" << std::endl;
    std::cout << "Shape: [";
    for (size_t i = 0; i < t.shape().dims.size(); ++i) {
        std::cout << t.shape().dims[i];
        if (i < t.shape().dims.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Stride: [";
    for (size_t i = 0; i < t.stride().strides.size(); ++i) {
        std::cout << t.stride().strides[i];
        if (i < t.stride().strides.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Dtype: " << get_dtype_name(t.dtype()) << std::endl;
    std::cout << "Device: " << (t.device().is_cpu() ? "CPU" : "CUDA") << std::endl;
    std::cout << "Numel: " << t.numel() << std::endl;
    std::cout << "Ndim: " << t.ndim() << std::endl;
    std::cout << "Requires Grad: " << (t.requires_grad() ? "true" : "false") << std::endl;
    std::cout << "Data: ";
    t.display();
}

int main() {
    std::cout << "Running FP4 Basic Ops Test..." << std::endl;
    
    // 1. Setup Data
    // A = [1.0, 2.0]
    // B = [0.5, 3.0]
    Tensor A(Shape{{2}}, Dtype::Float4_e2m1, Device::CPU);
    Tensor B(Shape{{2}}, Dtype::Float4_e2m1, Device::CPU);
    
    std::vector<float4_e2m1_t> data_a = {float4_e2m1_t(1.0f), float4_e2m1_t(2.0f)};
    std::vector<float4_e2m1_t> data_b = {float4_e2m1_t(0.5f), float4_e2m1_t(3.0f)};
    
    A.set_data(data_a);
    B.set_data(data_b);

    // Print inputs once
    std::cout << "\n========== INPUT TENSORS ==========" << std::endl;
    print_tensor_metadata("Tensor A", A);
    print_tensor_metadata("Tensor B", B);

    // 2. Addition
    // [1.0+0.5, 2.0+3.0] = [1.5, 5.0->6.0(saturate)]
    // Note: 2.0+3.0 = 5.0. Max FP4 is 6.0. 5.0 rounds to 6.0.
    Tensor C_add = A + B;

    if (check_tensor(C_add, {1.5f, 6.0f})) {
        std::cout << "\nAddition: PASS" << std::endl;
    } else {
        std::cout << "\nAddition: FAIL" << std::endl;
    }

    // 3. Subtraction
    // [1.0-0.5, 2.0-3.0] = [0.5, -1.0]
    Tensor C_sub = A - B;
    if (check_tensor(C_sub, {0.5f, -1.0f})) {
        std::cout << "\nSubtraction: PASS" << std::endl;
    } else {
        std::cout << "\nSubtraction: FAIL" << std::endl;
    }

    // 4. Multiplication
    // [1.0*0.5, 2.0*3.0] = [0.5, 6.0->6.0(saturate)]
    Tensor C_mul = A * B;
    if (check_tensor(C_mul, {0.5f, 6.0f})) {
        std::cout << "\nMultiplication: PASS" << std::endl;
    } else {
        std::cout << "\nMultiplication: FAIL" << std::endl;
    }

    // 5. Division
    // [1.0/0.5, 2.0/3.0] = [2.0, 0.666...]
    // 0.666... rounds to 0.5 (0.5) or 1.0 (1.0)?
    // 0.666 is closer to 0.5 (diff 0.166) than 1.0 (diff 0.333). -> 0.5.
    Tensor C_div = A / B;
    if (check_tensor(C_div, {2.0f, 0.5f})) {
        std::cout << "\nDivision: PASS" << std::endl;
    } else {
        std::cout << "\nDivision: FAIL" << std::endl;
    }

    // Print all result tensors with metadata
    std::cout << "\n========== RESULT TENSORS ==========" << std::endl;
    print_tensor_metadata("C_add (A + B)", C_add);
    print_tensor_metadata("C_sub (A - B)", C_sub);
    print_tensor_metadata("C_mul (A * B)", C_mul);
    print_tensor_metadata("C_div (A / B)", C_div);

    return 0;
}
