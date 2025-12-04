#include "TensorLib.h"
#include <iostream>

using namespace OwnTensor;

int main()
{
    std::cout << "Running FP4 Basic Ops Test..." << std::endl;  
    // 1. Setup Data
    // A = [1.0, 2.0]
    // B = [0.5, 3.0]
    Tensor A(Shape{{2}}, Dtype::Float4_e2m1_2x, Device::CPU);
    // Tensor B(Shape{{2}}, Dtype::Float4_e2m1, Device::CPU);
    
    std::vector<float4_e2m1_2x_t> data_a = {float4_e2m1_2x_t(1.0f), float4_e2m1_2x_t(2.510f)};
    // std::vector<float4_e2m1_t> data_b = {float4_e2m1_t(0.5f), float4_e2m1_t(3.0f)};
    
    A.set_data(data_a);
    // B.set_data(data_b);

    std::cout << "Input A\n";
    A.display();

} 