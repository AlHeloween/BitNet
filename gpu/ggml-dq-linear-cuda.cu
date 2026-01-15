#include <cuda_runtime.h>
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"
#include "../../lib/cuda_math.h"

using namespace DynaMap::math;

// DQLinear: Batch Hamilton product matrix multiplication
// Weight: [out_features, in_features, 8]
// Input: [batch, seq, in_features, 8]
// Output: [batch, seq, out_features, 8]

// For each output feature i: y_i = sum_j (W[i,j] * x_j)
__global__ void ggml_dq_linear_cuda(
    const float* weight,      // [out_features, in_features, 8]
    const float* input,        // [batch, seq, in_features, 8]
    float* output,             // [batch, seq, out_features, 8]
    int batch_size,
    int seq_len,
    int in_features,
    int out_features
) {
    // Each thread handles one output position: [batch, seq, out_feature]
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int out_feat = threadIdx.x;
    
    if (batch >= batch_size || seq >= seq_len || out_feat >= out_features) {
        return;
    }
    
    // Initialize output to zero dual quaternion
    float out_dq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Sum over input features: y_i = sum_j (W[i,j] * x_j)
    for (int j = 0; j < in_features; j++) {
        // Load weight: W[out_feat, j, :]
        int w_idx = (out_feat * in_features + j) * 8;
        Quaternion w_r(weight[w_idx + 0], weight[w_idx + 1], 
                      weight[w_idx + 2], weight[w_idx + 3]);
        Quaternion w_d(weight[w_idx + 4], weight[w_idx + 5], 
                      weight[w_idx + 6], weight[w_idx + 7]);
        dualQuat w_dq(w_r, w_d);
        
        // Load input: x[batch, seq, j, :]
        int x_idx = ((batch * seq_len + seq) * in_features + j) * 8;
        Quaternion x_r(input[x_idx + 0], input[x_idx + 1], 
                      input[x_idx + 2], input[x_idx + 3]);
        Quaternion x_d(input[x_idx + 4], input[x_idx + 5], 
                      input[x_idx + 6], input[x_idx + 7]);
        dualQuat x_dq(x_r, x_d);
        
        // Hamilton product: W[out_feat, j] * x_j
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
    
    // Store output
    int out_idx = ((batch * seq_len + seq) * out_features + out_feat) * 8;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        output[out_idx + k] = out_dq[k];
    }
}

// Host wrapper
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
    ) {
        dim3 blocks(batch_size, seq_len);
        int threads = out_features;
        ggml_dq_linear_cuda<<<blocks, threads, 0, stream>>>(
            weight, input, output, batch_size, seq_len, in_features, out_features
        );
    }
}
