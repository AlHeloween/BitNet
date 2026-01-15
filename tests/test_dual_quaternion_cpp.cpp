#include "ggml-dual-quaternion.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>

// Test Hamilton product correctness
void test_hamilton_product() {
    std::cout << "Testing dual quaternion Hamilton product...\n";
    
    // Create GGML context
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,  // 1MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create test dual quaternions
    // q1 = identity rotation + zero translation
    float q1_data[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // q2 = identity rotation + zero translation
    float q2_data[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    int64_t ne[1] = {8};
    struct ggml_tensor* q1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    struct ggml_tensor* q2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    memcpy(q1->data, q1_data, 8 * sizeof(float));
    memcpy(q2->data, q2_data, 8 * sizeof(float));
    
    // Compute Hamilton product
    struct ggml_tensor* result = ggml_dual_quaternion_hamilton_product(ctx, q1, q2);
    
    // Verify result (identity * identity = identity)
    const float* result_data = (const float*)result->data;
    assert(fabs(result_data[0] - 1.0f) < 1e-6f);  // w_r should be 1
    for (int i = 1; i < 8; i++) {
        assert(fabs(result_data[i]) < 1e-6f);  // All other components should be 0
    }
    
    ggml_free(ctx);
    std::cout << "  ✓ Hamilton product works\n";
}

// Test conjugate
void test_conjugate() {
    std::cout << "Testing dual quaternion conjugate...\n";
    
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create test dual quaternion: q = [1, 2, 3, 4, 5, 6, 7, 8]
    float q_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    struct ggml_tensor* q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    memcpy(q->data, q_data, 8 * sizeof(float));
    
    // Compute conjugate
    struct ggml_tensor* result = ggml_dual_quaternion_conjugate(ctx, q);
    
    // Verify: conjugate should negate x, y, z of both rotation and translation
    const float* result_data = (const float*)result->data;
    assert(fabs(result_data[0] - 1.0f) < 1e-6f);   // w_r unchanged
    assert(fabs(result_data[1] - (-2.0f)) < 1e-6f);  // x_r negated
    assert(fabs(result_data[2] - (-3.0f)) < 1e-6f);  // y_r negated
    assert(fabs(result_data[3] - (-4.0f)) < 1e-6f);  // z_r negated
    assert(fabs(result_data[4] - 5.0f) < 1e-6f);     // w_d unchanged
    assert(fabs(result_data[5] - (-6.0f)) < 1e-6f);  // x_d negated
    assert(fabs(result_data[6] - (-7.0f)) < 1e-6f);  // y_d negated
    assert(fabs(result_data[7] - (-8.0f)) < 1e-6f);  // z_d negated
    
    ggml_free(ctx);
    std::cout << "  ✓ Conjugate works\n";
}

// Test normalize
void test_normalize() {
    std::cout << "Testing dual quaternion normalize...\n";
    
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create test dual quaternion with non-unit rotation
    float q_data[8] = {2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    struct ggml_tensor* q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    memcpy(q->data, q_data, 8 * sizeof(float));
    
    // Normalize
    struct ggml_tensor* result = ggml_dual_quaternion_normalize(ctx, q);
    
    // Verify: rotation quaternion should be normalized
    const float* result_data = (const float*)result->data;
    float rot_norm = sqrtf(result_data[0]*result_data[0] + result_data[1]*result_data[1] + 
                           result_data[2]*result_data[2] + result_data[3]*result_data[3]);
    assert(fabs(rot_norm - 1.0f) < 1e-6f);
    
    ggml_free(ctx);
    std::cout << "  ✓ Normalize works\n";
}

// Test inverse
void test_inverse() {
    std::cout << "Testing dual quaternion inverse...\n";
    
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create unit dual quaternion
    float q_data[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    struct ggml_tensor* q = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    memcpy(q->data, q_data, 8 * sizeof(float));
    
    // Compute inverse
    struct ggml_tensor* inv = ggml_dual_quaternion_inverse(ctx, q);
    
    // Compute q * inv (should be identity)
    struct ggml_tensor* product = ggml_dual_quaternion_hamilton_product(ctx, q, inv);
    
    // Verify product is identity
    const float* product_data = (const float*)product->data;
    assert(fabs(product_data[0] - 1.0f) < 1e-5f);
    for (int i = 1; i < 8; i++) {
        assert(fabs(product_data[i]) < 1e-5f);
    }
    
    ggml_free(ctx);
    std::cout << "  ✓ Inverse works\n";
}

int main() {
    std::cout << "=== Dual Quaternion C++ Tests ===\n\n";
    
    test_hamilton_product();
    test_conjugate();
    test_normalize();
    test_inverse();
    
    std::cout << "\n✓ All dual quaternion tests passed!\n";
    return 0;
}
