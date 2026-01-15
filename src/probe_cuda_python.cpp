// Standalone C++ CUDA probe library for Python bindings
// This provides a clean interface for calling C++ CUDA probe operations from Python

#include <cuda_runtime.h>
#include <vector>
#include <cstring>

// Include CUDA kernels
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
    
    void ggml_dq_attention_scores_cuda_launch(
        const float* q,
        const float* k,
        float* scores,
        int batch_size,
        int seq_q,
        int seq_k,
        float semantic_weight,
        float contextual_weight,
        cudaStream_t stream
    );
}

// Python-friendly wrapper functions
extern "C" {

// DQLinear forward pass
// Returns 0 on success, non-zero on error
int probe_cuda_dq_linear(
    const float* weight,      // [out_features, in_features, 8]
    const float* input,        // [batch, seq, in_features, 8]
    float* output,             // [batch, seq, out_features, 8]
    int batch_size,
    int seq_len,
    int in_features,
    int out_features
) {
    cudaError_t err;
    cudaStream_t stream = 0;  // Default stream
    
    // Launch CUDA kernel
    ggml_dq_linear_cuda_launch(
        weight, input, output,
        batch_size, seq_len, in_features, out_features,
        stream
    );
    
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return 2;
    }
    
    return 0;
}

// DQAttention forward pass
// Returns 0 on success, non-zero on error
int probe_cuda_dq_attention_scores(
    const float* q,           // [batch, seq_q, 8]
    const float* k,           // [batch, seq_k, 8]
    float* scores,            // [batch, seq_q, seq_k]
    int batch_size,
    int seq_q,
    int seq_k,
    float semantic_weight,
    float contextual_weight
) {
    cudaError_t err;
    cudaStream_t stream = 0;  // Default stream
    
    // Launch CUDA kernel
    ggml_dq_attention_scores_cuda_launch(
        q, k, scores,
        batch_size, seq_q, seq_k,
        semantic_weight, contextual_weight,
        stream
    );
    
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return 2;
    }
    
    return 0;
}

}  // extern "C"
