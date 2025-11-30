#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "core/Tensor.h"
#include "ops/TensorOps.h"
#include "dtype/Types.h"

using namespace OwnTensor;

bool check_tensor_val(Tensor& t, std::vector<float> expected) {
    if (t.numel() != expected.size()) return false;
    
    // Convert to float for checking
    Tensor t_float = t.as_type(Dtype::Float32);
    const float* data = t_float.data<float>();
    
    for (size_t i = 0; i < t.numel(); ++i) {
        if (std::isnan(expected[i])) {
            if (!std::isnan(data[i])) return false;
        } else if (std::isinf(expected[i])) {
            if (!std::isinf(data[i])) return false;
            if ((expected[i] > 0) != (data[i] > 0)) return false;
        } else {
            if (data[i] != expected[i]) return false;
        }
    }
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "TEST: FP4 Tensor Operations" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int total = 0;

    // 1. Creation and Fill
    total++;
    std::cout << "Test 1: Creation and Fill... ";
    Tensor t1(Shape{{2, 2}}, Dtype::Float4_e2m1);
    std::vector<float4_e2m1_t> data1 = {
        float4_e2m1_t(1.0f), float4_e2m1_t(2.0f),
        float4_e2m1_t(0.5f), float4_e2m1_t(1.5f)
    };
    t1.set_data(data1);
    
    if (check_tensor_val(t1, {1.0f, 2.0f, 0.5f, 1.5f})) {
        std::cout << "[PASS]" << std::endl;
        passed++;
    } else {
        std::cout << "[FAIL]" << std::endl;
    }

    // 2. Addition (FP4 + FP4)
    total++;
    std::cout << "Test 2: Addition (FP4 + FP4)... ";
    Tensor t2(Shape{{2, 2}}, Dtype::Float4_e2m1);
    std::vector<float4_e2m1_t> data2 = {
        float4_e2m1_t(2.0f), float4_e2m1_t(1.0f),
        float4_e2m1_t(0.5f), float4_e2m1_t(1.5f)
    };
    t2.set_data(data2);

    // 1+2=3, 2+1=3, 0.5+0.5=1.0, 1.5+1.5=3.0
    Tensor t_add = t1 + t2; 
    // Result promotes to Float32 by default as per DtypeTraits
    // But we can cast back to FP4 to check storage
    Tensor t_add_fp4 = t_add.as_type(Dtype::Float4_e2m1);

    if (check_tensor_val(t_add_fp4, {3.0f, 3.0f, 1.0f, 3.0f})) {
        std::cout << "[PASS]" << std::endl;
        passed++;
    } else {
        std::cout << "[FAIL]" << std::endl;
        t_add.display();
    }

    // 3. Subtraction
    total++;
    std::cout << "Test 3: Subtraction... ";
    // 1-2=-1, 2-1=1, 0.5-0.5=0, 1.5-1.5=0
    Tensor t_sub = t1 - t2;
    Tensor t_sub_fp4 = t_sub.as_type(Dtype::Float4_e2m1);
    
    if (check_tensor_val(t_sub_fp4, {-1.0f, 1.0f, 0.0f, 0.0f})) {
        std::cout << "[PASS]" << std::endl;
        passed++;
    } else {
        std::cout << "[FAIL]" << std::endl;
        t_sub.display();
    }

    // 4. Multiplication
    total++;
    std::cout << "Test 4: Multiplication... ";
    // 1*2=2, 2*1=2, 0.5*0.5=0.25->0.5(round up) or 0(round down)? 
    // 0.25 is midpoint. Code: if < 0.25 -> 0. else -> 0.5.
    // So 0.25 -> 0.5.
    // 1.5*1.5=2.25 -> 2.0 (nearest to 2.0 or 3.0? 2.0 is 2.0, 3.0 is 3.0. 2.25 is closer to 2.0)
    Tensor t_mul = t1 * t2;
    Tensor t_mul_fp4 = t_mul.as_type(Dtype::Float4_e2m1);

    if (check_tensor_val(t_mul_fp4, {2.0f, 2.0f, 0.5f, 2.0f})) {
        std::cout << "[PASS]" << std::endl;
        passed++;
    } else {
        std::cout << "[FAIL]" << std::endl;
        t_mul.display();
    }

    // 5. Division
    total++;
    std::cout << "Test 5: Division... ";
    // 1/2=0.5, 2/1=2, 0.5/0.5=1, 1.5/1.5=1
    Tensor t_div = t1 / t2;
    Tensor t_div_fp4 = t_div.as_type(Dtype::Float4_e2m1);

    if (check_tensor_val(t_div_fp4, {0.5f, 2.0f, 1.0f, 1.0f})) {
        std::cout << "[PASS]" << std::endl;
        passed++;
    } else {
        std::cout << "[FAIL]" << std::endl;
        t_div.display();
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;

    return (passed == total) ? 0 : 1;
}
