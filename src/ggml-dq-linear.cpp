#include "ggml-dq-linear.h"
#include "ggml-dual-quaternion.h"
#include "ggml.h"
#include "../../../lib/math_dualQuat.h"
#include "../../../lib/math_Quat.h"

#include <cmath>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
// Forward declaration for CUDA kernel launch
extern "C" {
    void ggml_dq_linear_cuda_launch(
        const float* weight,
        const float* input,
        float* output,
        int batch_size,
        int seq_len,
        int in_features,
        int out_features,
        cudaStream_t stream
    );
}
#endif

using namespace DynaMap::math;

struct ggml_tensor* ggml_dq_linear(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* weight,
    struct ggml_tensor* bias
) {
    // x: [batch, seq, in_features, 8]
    // weight: [out_features, in_features, 8]
    // bias: [out_features, 8] or NULL
    // output: [batch, seq, out_features, 8]
    
    GGML_ASSERT(ggml_n_dims(x) == 4);
    GGML_ASSERT(x->ne[3] == 8);  // Last dim must be 8 (dual quaternion)
    GGML_ASSERT(ggml_n_dims(weight) == 3);
    GGML_ASSERT(weight->ne[2] == 8);
    GGML_ASSERT(x->ne[2] == weight->ne[1]);  // in_features must match
    
    int batch_size = x->ne[0];
    int seq_len = x->ne[1];
    int in_features = x->ne[2];
    int out_features = weight->ne[0];
    
    // Create output tensor: [batch, seq, out_features, 8]
    int64_t out_ne[4] = {batch_size, seq_len, out_features, 8};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, out_ne);
    
    const float* x_data = (const float*)x->data;
    const float* weight_data = (const float*)weight->data;
    float* out_data = (float*)result->data;
    
    // Check if we should use CUDA backend
    bool use_cuda = false;
#ifdef GGML_USE_CUDA
    // Check if tensors are on CUDA backend
    if (x->backend != GGML_BACKEND_TYPE_CPU || 
        weight->backend != GGML_BACKEND_TYPE_CPU) {
        use_cuda = true;
    }
    // Also check if CUDA is available at runtime
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        use_cuda = true;  // CUDA available, prefer GPU
    }
#endif
    
    if (use_cuda) {
#ifdef GGML_USE_CUDA
        // Use CUDA kernel for DQLinear computation
        cudaStream_t stream = 0;  // Default stream
        ggml_dq_linear_cuda_launch(
            weight_data, x_data, out_data,
            batch_size, seq_len, in_features, out_features,
            stream
        );
        // Synchronize to ensure computation is complete
        cudaStreamSynchronize(stream);
        
        // Add bias on CPU if present (bias is typically small, CPU is fine)
        if (bias) {
            const float* b_data = (const float*)bias->data;
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < seq_len; s++) {
                    for (int o = 0; o < out_features; o++) {
                        int out_idx = ((b * seq_len + s) * out_features + o) * 8;
                        const float* b_o = &b_data[o * 8];
                        for (int k = 0; k < 8; k++) {
                            out_data[out_idx + k] += b_o[k];
                        }
                    }
                }
            }
        }
#else
        use_cuda = false;  // Fallback to CPU if CUDA not compiled
#endif
    }
    
    if (!use_cuda) {
        // CPU implementation
        // For each output position: y[batch, seq, out_feat] = sum_j (W[out_feat, j] * x[batch, seq, j])
        for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_features; o++) {
                // Initialize output to zero dual quaternion
                float out_dq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                
                // Sum over input features: y_i = sum_j (W[i, j] * x_j)
                for (int j = 0; j < in_features; j++) {
                    // Load weight: W[o, j, :]
                    const float* w = &weight_data[(o * in_features + j) * 8];
                    Quaternion w_r(w[0], w[1], w[2], w[3]);
                    Quaternion w_d(w[4], w[5], w[6], w[7]);
                    dualQuat w_dq(w_r, w_d);
                    
                    // Load input: x[b, s, j, :]
                    const float* x_j = &x_data[((b * seq_len + s) * in_features + j) * 8];
                    Quaternion x_r(x_j[0], x_j[1], x_j[2], x_j[3]);
                    Quaternion x_d(x_j[4], x_j[5], x_j[6], x_j[7]);
                    dualQuat x_dq(x_r, x_d);
                    
                    // Hamilton product: W[o, j] * x_j
                    dualQuat prod = dualQuat::mul(x_dq, w_dq);
                    
                    // Accumulate
                    out_dq[0] += prod.real.w;
                    out_dq[1] += prod.real.x;
                    out_dq[2] += prod.real.y;
                    out_dq[3] += prod.real.z;
                    out_dq[4] += prod.dual.w;
                    out_dq[5] += prod.dual.x;
                    out_dq[6] += prod.dual.y;
                    out_dq[7] += prod.dual.z;
                }
                
                // Add bias if present
                if (bias) {
                    const float* b_data = (const float*)bias->data;
                    const float* b = &b_data[o * 8];
                    for (int k = 0; k < 8; k++) {
                        out_dq[k] += b[k];
                    }
                }
                
                // Store output
                float* out = &out_data[((b * seq_len + s) * out_features + o) * 8];
                for (int k = 0; k < 8; k++) {
                    out[k] = out_dq[k];
                }
            }
        }
    }
    }  // End of CPU implementation
    
    return result;
}
