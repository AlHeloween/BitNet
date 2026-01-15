#include "ggml-dq-linear.h"
#include "ggml-dual-quaternion.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>

// Test DQLinear forward pass
void test_dq_linear_forward() {
    std::cout << "Testing DQLinear forward pass...\n";
    
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create test input: [batch=1, seq=2, in_features=3, 8]
    int batch = 1, seq = 2, in_features = 3;
    int64_t x_ne[4] = {batch, seq, in_features, 8};
    struct ggml_tensor* x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, x_ne);
    
    // Initialize input with identity dual quaternions
    float* x_data = (float*)x->data;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int f = 0; f < in_features; f++) {
                int idx = ((b * seq + s) * in_features + f) * 8;
                x_data[idx + 0] = 1.0f;  // w_r
                for (int i = 1; i < 8; i++) {
                    x_data[idx + i] = 0.0f;
                }
            }
        }
    }
    
    // Create weights: [out_features=2, in_features=3, 8]
    int out_features = 2;
    int64_t w_ne[3] = {out_features, in_features, 8};
    struct ggml_tensor* weight = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, w_ne);
    
    // Initialize weights with identity dual quaternions
    float* w_data = (float*)weight->data;
    for (int o = 0; o < out_features; o++) {
        for (int i = 0; i < in_features; i++) {
            int idx = (o * in_features + i) * 8;
            w_data[idx + 0] = 1.0f;  // w_r
            for (int j = 1; j < 8; j++) {
                w_data[idx + j] = 0.0f;
            }
        }
    }
    
    // Forward pass
    struct ggml_tensor* output = ggml_dq_linear(ctx, x, weight, NULL);
    
    // Verify output shape: [batch, seq, out_features, 8]
    assert(output->n_dims == 4);
    assert(output->ne[0] == batch);
    assert(output->ne[1] == seq);
    assert(output->ne[2] == out_features);
    assert(output->ne[3] == 8);
    
    // Verify output values (identity * identity = identity)
    const float* out_data = (const float*)output->data;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int o = 0; o < out_features; o++) {
                int idx = ((b * seq + s) * out_features + o) * 8;
                assert(fabs(out_data[idx + 0] - 1.0f) < 1e-5f);  // w_r should be 1
                for (int i = 1; i < 8; i++) {
                    assert(fabs(out_data[idx + i]) < 1e-5f);  // Other components should be 0
                }
            }
        }
    }
    
    ggml_free(ctx);
    std::cout << "  ✓ DQLinear forward pass works\n";
}

// Test with bias
void test_dq_linear_with_bias() {
    std::cout << "Testing DQLinear with bias...\n";
    
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create minimal test case
    int batch = 1, seq = 1, in_features = 1, out_features = 1;
    int64_t x_ne[4] = {batch, seq, in_features, 8};
    struct ggml_tensor* x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, x_ne);
    
    // Zero input
    memset(x->data, 0, batch * seq * in_features * 8 * sizeof(float));
    
    // Identity weight
    int64_t w_ne[3] = {out_features, in_features, 8};
    struct ggml_tensor* weight = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, w_ne);
    float* w_data = (float*)weight->data;
    w_data[0] = 1.0f;
    for (int i = 1; i < 8; i++) w_data[i] = 0.0f;
    
    // Identity bias
    int64_t b_ne[2] = {out_features, 8};
    struct ggml_tensor* bias = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, b_ne);
    float* b_data = (float*)bias->data;
    b_data[0] = 1.0f;
    for (int i = 1; i < 8; i++) b_data[i] = 0.0f;
    
    // Forward pass
    struct ggml_tensor* output = ggml_dq_linear(ctx, x, weight, bias);
    
    // Verify output (zero input * identity weight + identity bias = identity)
    const float* out_data = (const float*)output->data;
    assert(fabs(out_data[0] - 1.0f) < 1e-5f);
    for (int i = 1; i < 8; i++) {
        assert(fabs(out_data[i]) < 1e-5f);
    }
    
    ggml_free(ctx);
    std::cout << "  ✓ DQLinear with bias works\n";
}

int main() {
    std::cout << "=== DQLinear C++ Tests ===\n\n";
    
    test_dq_linear_forward();
    test_dq_linear_with_bias();
    
    std::cout << "\n✓ All DQLinear tests passed!\n";
    return 0;
}
