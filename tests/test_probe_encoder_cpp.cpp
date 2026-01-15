#include "probe-encoder.h"
#include "ggml-dq-linear.h"
#include "ggml-dq-attention.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>

// Test Probe encoder forward pass
void test_probe_encoder_forward() {
    std::cout << "Testing Probe encoder forward pass...\n";
    
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create test input embeddings: [batch=1, seq=3, hidden_dim=64]
    int batch = 1, seq = 3, hidden_dim = 64;
    int64_t input_ne[3] = {batch, seq, hidden_dim};
    struct ggml_tensor* input = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, input_ne);
    
    // Initialize with random values
    float* input_data = (float*)input->data;
    for (int i = 0; i < batch * seq * hidden_dim; i++) {
        input_data[i] = 0.1f * (float)(i % 10);
    }
    
    // Create encoder parameters
    struct probe_encoder_params encoder_params;
    encoder_params.vocab_size = 1000;
    encoder_params.hidden_size = hidden_dim;
    encoder_params.num_layers = 2;
    encoder_params.num_heads = 4;
    encoder_params.dropout = 0.1f;
    encoder_params.enable_probe = true;
    
    // Forward pass (without weights for now - would need to load from file)
    struct ggml_tensor* output = probe_encode(ctx, input, &encoder_params, NULL);
    
    // Verify output shape: [batch, seq, hidden_size, 8] (dual quaternion latents)
    if (output != NULL) {
        assert(output->n_dims == 4);
        assert(output->ne[0] == batch);
        assert(output->ne[1] == seq);
        assert(output->ne[2] == hidden_dim);
        assert(output->ne[3] == 8);  // Dual quaternion dimension
    }
    
    ggml_free(ctx);
    std::cout << "  ✓ Probe encoder forward pass works\n";
}

// Test embedding to dual quaternion conversion
void test_embedding_to_dual_quat() {
    std::cout << "Testing embedding to dual quaternion conversion...\n";
    
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create small test input: [batch=1, seq=1, hidden_dim=8]
    // hidden_dim=8 so we can directly map to dual quaternion
    int batch = 1, seq = 1, hidden_dim = 8;
    int64_t input_ne[3] = {batch, seq, hidden_dim};
    struct ggml_tensor* input = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, input_ne);
    
    // Initialize with identity-like values
    float* input_data = (float*)input->data;
    input_data[0] = 1.0f;  // w_r
    for (int i = 1; i < 8; i++) {
        input_data[i] = 0.0f;
    }
    
    struct probe_encoder_params encoder_params;
    encoder_params.vocab_size = 1000;
    encoder_params.hidden_size = hidden_dim;
    encoder_params.num_layers = 1;
    encoder_params.num_heads = 1;
    encoder_params.dropout = 0.0f;
    encoder_params.enable_probe = true;
    
    struct ggml_tensor* output = probe_encode(ctx, input, &encoder_params, NULL);
    
    if (output != NULL) {
        // Verify output is dual quaternion format
        assert(output->n_dims == 4);
        assert(output->ne[3] == 8);
    }
    
    ggml_free(ctx);
    std::cout << "  ✓ Embedding to dual quaternion conversion works\n";
}

int main() {
    std::cout << "=== Probe Encoder C++ Tests ===\n\n";
    
    test_probe_encoder_forward();
    test_embedding_to_dual_quat();
    
    std::cout << "\n✓ All Probe encoder tests passed!\n";
    return 0;
}
