#include <cuda_runtime.h>
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"
#include "../../lib/cuda_math.h"

using namespace DynaMap::math;

// CUDA kernel: Triplet synthesis (mean)
__global__ void ggml_transcender_synthesize_mean_cuda(
    const float* entry_a,  // [8]
    const float* entry_b,  // [8]
    const float* entry_c,  // [8]
    float* out,            // [8]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int i = idx % 8;
    int batch = idx / 8;
    
    const float* a = &entry_a[batch * 8];
    const float* b = &entry_b[batch * 8];
    const float* c = &entry_c[batch * 8];
    float* o = &out[batch * 8];
    
    o[i] = (a[i] + b[i] + c[i]) / 3.0f;
}

// CUDA kernel: Triplet synthesis (max)
__global__ void ggml_transcender_synthesize_max_cuda(
    const float* entry_a,
    const float* entry_b,
    const float* entry_c,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int i = idx % 8;
    int batch = idx / 8;
    
    const float* a = &entry_a[batch * 8];
    const float* b = &entry_b[batch * 8];
    const float* c = &entry_c[batch * 8];
    float* o = &out[batch * 8];
    
    o[i] = fmaxf(fmaxf(a[i], b[i]), c[i]);
}

// Host wrapper functions
extern "C" {
    void ggml_transcender_synthesize_mean_cuda_launch(
        const float* entry_a,
        const float* entry_b,
        const float* entry_c,
        float* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_transcender_synthesize_mean_cuda<<<blocks, threads, 0, stream>>>(
            entry_a, entry_b, entry_c, out, n
        );
    }
    
    void ggml_transcender_synthesize_max_cuda_launch(
        const float* entry_a,
        const float* entry_b,
        const float* entry_c,
        float* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_transcender_synthesize_max_cuda<<<blocks, threads, 0, stream>>>(
            entry_a, entry_b, entry_c, out, n
        );
    }
}
