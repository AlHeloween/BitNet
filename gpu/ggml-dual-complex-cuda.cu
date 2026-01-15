#include <cuda_runtime.h>
#include "../../lib/math_dualPhaser.h"
#include "../../lib/cuda_math.h"

using namespace DynaMap::math;

// Kernel: Dual-complex addition
__global__ void ggml_dual_complex_add_cuda(
    const float4* a,      // Packed dual phaser [real.x, real.y, dual.x, dual.y]
    const float4* b,
    float4* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Unpack using existing function
    dualPhaser da = dualphaser_unpack(a[idx]);
    dualPhaser db = dualphaser_unpack(b[idx]);
    
    // Use existing addition
    dualPhaser result = dualphaser_add(da, db);
    
    // Pack using existing function
    out[idx] = dualphaser_pack(result);
}

// Kernel: Dual-complex multiplication
__global__ void ggml_dual_complex_mul_cuda(
    const float4* a,
    const float4* b,
    float4* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    dualPhaser da = dualphaser_unpack(a[idx]);
    dualPhaser db = dualphaser_unpack(b[idx]);
    
    // Use existing multiplication
    dualPhaser result = dualphaser_mul(da, db);
    
    out[idx] = dualphaser_pack(result);
}

// Kernel: Dual-complex conjugate
__global__ void ggml_dual_complex_conjugate_cuda(
    const float4* a,
    float4* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    dualPhaser da = dualphaser_unpack(a[idx]);
    dualPhaser result = dualphaser_conjugate(da);
    
    out[idx] = dualphaser_pack(result);
}

// Kernel: Dual-complex normalize
__global__ void ggml_dual_complex_normalize_cuda(
    const float4* a,
    float4* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    dualPhaser da = dualphaser_unpack(a[idx]);
    dualPhaser result = dualphaser_normalize(da);
    
    out[idx] = dualphaser_pack(result);
}

// Host wrapper functions
extern "C" {
    void ggml_dual_complex_add_cuda_launch(
        const float4* a,
        const float4* b,
        float4* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_dual_complex_add_cuda<<<blocks, threads, 0, stream>>>(a, b, out, n);
    }
    
    void ggml_dual_complex_mul_cuda_launch(
        const float4* a,
        const float4* b,
        float4* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_dual_complex_mul_cuda<<<blocks, threads, 0, stream>>>(a, b, out, n);
    }
    
    void ggml_dual_complex_conjugate_cuda_launch(
        const float4* a,
        float4* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_dual_complex_conjugate_cuda<<<blocks, threads, 0, stream>>>(a, out, n);
    }
    
    void ggml_dual_complex_normalize_cuda_launch(
        const float4* a,
        float4* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_dual_complex_normalize_cuda<<<blocks, threads, 0, stream>>>(a, out, n);
    }
}
