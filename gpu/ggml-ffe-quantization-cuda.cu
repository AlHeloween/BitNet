#include <cuda_runtime.h>
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"
#include "../../lib/cuda_math.h"

using namespace DynaMap::math;

// CUDA kernel: FFE quantization
// Find nearest centroid for each dual quaternion
__global__ void ggml_ffe_quantize_cuda(
    const float* dual_quaternions,  // [n, 8]
    const float* centroids,         // [n_levels, n_per_level, 8]
    int32_t* addresses_out,         // [n] (14-bit integers)
    int n,
    int n_levels,
    int n_per_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Load dual quaternion
    const float* dq = &dual_quaternions[idx * 8];
    Quaternion dq_r(dq[0], dq[1], dq[2], dq[3]);
    Quaternion dq_d(dq[4], dq[5], dq[6], dq[7]);
    
    // Find nearest centroid
    int best_level = 0;
    int best_index = 0;
    float best_distance = 1e9f;
    
    for (int level = 0; level < n_levels; level++) {
        for (int idx_c = 0; idx_c < n_per_level; idx_c++) {
            const float* centroid = &centroids[(level * n_per_level + idx_c) * 8];
            
            // Compute L2 distance
            float dist = 0.0f;
            for (int j = 0; j < 8; j++) {
                float diff = dq[j] - centroid[j];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
            
            if (dist < best_distance) {
                best_distance = dist;
                best_level = level;
                best_index = idx_c;
            }
        }
    }
    
    // Encode FFE address (3-9-2 format)
    uint16_t addr_bits = (best_level << 11) | (best_index << 2) | 0;
    addresses_out[idx] = (int32_t)addr_bits;
}

// CUDA kernel: FFE dequantization
// Get centroids for addresses
__global__ void ggml_ffe_dequantize_cuda(
    const int32_t* addresses,       // [n] (14-bit integers)
    const float* centroids,         // [n_levels, n_per_level, 8]
    float* dual_quaternions_out,    // [n, 8]
    int n,
    int n_per_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Decode address
    uint16_t addr_bits = (uint16_t)addresses[idx];
    uint8_t level = (addr_bits >> 11) & 0x7;
    uint16_t index = (addr_bits >> 2) & 0x1FF;
    
    // Get centroid
    const float* centroid = &centroids[(level * n_per_level + index) * 8];
    float* out = &dual_quaternions_out[idx * 8];
    
    // Copy to output
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        out[j] = centroid[j];
    }
}

// Host wrapper functions
extern "C" {
    void ggml_ffe_quantize_cuda_launch(
        const float* dual_quaternions,
        const float* centroids,
        int32_t* addresses_out,
        int n,
        int n_levels,
        int n_per_level,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_ffe_quantize_cuda<<<blocks, threads, 0, stream>>>(
            dual_quaternions, centroids, addresses_out, n, n_levels, n_per_level
        );
    }
    
    void ggml_ffe_dequantize_cuda_launch(
        const int32_t* addresses,
        const float* centroids,
        float* dual_quaternions_out,
        int n,
        int n_per_level,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_ffe_dequantize_cuda<<<blocks, threads, 0, stream>>>(
            addresses, centroids, dual_quaternions_out, n, n_per_level
        );
    }
}
