#include <iostream>
#include <vector>
#include "core/Tensor.h"
#include "dtype/Types.h"

using namespace OwnTensor;

int main() {
    std::cout << "Running FP4 Casting Test..." << std::endl;

    // 1. FP32 -> FP4
    Tensor t_fp32(Shape{{4}}, Dtype::Float32);
    t_fp32.set_data<float>({0.5f, 1.0f, 3.0f, 4.0f}); // 4.0 -> Inf
    
    Tensor t_fp4 = t_fp32.as_type(Dtype::Float4_e2m1);
    
    // Verify FP4 data
    std::vector<float4_e2m1_t> expected_fp4 = {
        float4_e2m1_t(0.5f), float4_e2m1_t(1.0f), 
        float4_e2m1_t(3.0f), float4_e2m1_t(std::numeric_limits<float>::infinity())
    };
    
    const float4_e2m1_t* data_fp4 = t_fp4.data<float4_e2m1_t>();
    bool pass1 = true;
    for(int i=0; i<4; ++i) {
        if (static_cast<float>(data_fp4[i]) != static_cast<float>(expected_fp4[i])) pass1 = false;
    }
    std::cout << "FP32 -> FP4: " << (pass1 ? "PASS" : "FAIL") << std::endl;

    // 2. FP4 -> FP32
    Tensor t_back_fp32 = t_fp4.as_type(Dtype::Float32);
    const float* data_back = t_back_fp32.data<float>();
    bool pass2 = true;
    if (data_back[0] != 0.5f) pass2 = false;
    if (data_back[1] != 1.0f) pass2 = false;
    if (data_back[2] != 3.0f) pass2 = false;
    if (!std::isinf(data_back[3])) pass2 = false;
    std::cout << "FP4 -> FP32: " << (pass2 ? "PASS" : "FAIL") << std::endl;

    // 3. FP4 -> Int32
    // 0.5 -> 0, 1.0 -> 1, 3.0 -> 3, Inf -> Undefined (usually min int or max int)
    Tensor t_int32 = t_fp4.as_type(Dtype::Int32);
    const int32_t* data_int = t_int32.data<int32_t>();
    bool pass3 = true;
    if (data_int[0] != 0) pass3 = false;
    if (data_int[1] != 1) pass3 = false;
    if (data_int[2] != 3) pass3 = false;
    std::cout << "FP4 -> Int32: " << (pass3 ? "PASS" : "FAIL") << std::endl;

    return 0;
}
