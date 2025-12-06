#include "TensorLib.h"

using namespace OwnTensor;

int main()
{
    return 0;

    Tensor A({{6}}, Dtype::Float32);
    A.display();
    // std::vector<float4_e2m1_t> data = {float4_e2m1_t(1.5f), float4_e2m1_t(-2.455f), 
    //     float4_e2m1_t(.55f), float4_e2m1_t(-1.5f), 
    //     float4_e2m1_t(8.f),float4_e2m1_t(0.f)};
    // A.set_data(data);
    // return 1;
}


