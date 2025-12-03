#include <iostream>
#include <vector>
#include "core/Tensor.h"
#include "ops/TensorOps.h"
#include "dtype/Types.h"

using namespace OwnTensor;

int main() {
    std::cout << "Running FP4 Broadcasting Test..." << std::endl;

    // A: Shape [2, 2] -> [[1.0, 1.0], [1.0, 1.0]]
    Tensor A = Tensor::full(Shape{{2, 2}}, TensorOptions().with_dtype(Dtype::Float4_e2m1).with_device(DeviceIndex(Device::CPU)), 1.0f);
    
    // B: Shape [2, 1] -> [[1.0], [2.0]]
    Tensor B(Shape{{2, 1}}, Dtype::Float4_e2m1);
    std::vector<float4_e2m1_t> data_b = {float4_e2m1_t(1.0f), float4_e2m1_t(2.0f)};
    B.set_data(data_b);

    // Result should be [2, 2]:
    // [[1.0+1.0, 1.0+1.0], 
    //  [1.0+2.0, 1.0+2.0]]
    // = [[2.0, 2.0], [3.0, 3.0]]

    Tensor C = A + B;
    
    if (C.shape().dims[0] != 2 || C.shape().dims[1] != 2) {
        std::cout << "Shape Mismatch: FAIL" << std::endl;
        return 1;
    }

    Tensor C_fp32 = C.as_type(Dtype::Float32);
    const float* data = C_fp32.data<float>();
    
    bool pass = true;
    if (data[0] != 2.0f) pass = false;
    if (data[1] != 2.0f) pass = false;
    if (data[2] != 3.0f) pass = false;
    if (data[3] != 3.0f) pass = false;

    std::cout << "Broadcasting Add: " << (pass ? "PASS" : "FAIL") << std::endl;

    return 0;
}
