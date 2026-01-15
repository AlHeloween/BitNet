#include <cuda_runtime.h>
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"
#include "../../lib/cuda_math.h"

using namespace DynaMap::math;

// Kernel: Hamilton product for dual quaternions
__global__ void ggml_dual_quaternion_hamilton_product_cuda(
    const float* q1,      // [n, 8]
    const float* q2,      // [n, 8]
    float* out,           // [n, 8]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Load quaternions: [w, x, y, z]
    Quaternion q1_r(q1[idx * 8 + 0], q1[idx * 8 + 1], 
                   q1[idx * 8 + 2], q1[idx * 8 + 3]);
    Quaternion q1_d(q1[idx * 8 + 4], q1[idx * 8 + 5], 
                   q1[idx * 8 + 6], q1[idx * 8 + 7]);
    dualQuat dq1(q1_r, q1_d);
    
    Quaternion q2_r(q2[idx * 8 + 0], q2[idx * 8 + 1], 
                   q2[idx * 8 + 2], q2[idx * 8 + 3]);
    Quaternion q2_d(q2[idx * 8 + 4], q2[idx * 8 + 5], 
                   q2[idx * 8 + 6], q2[idx * 8 + 7]);
    dualQuat dq2(q2_r, q2_d);
    
    // Use existing Hamilton product
    dualQuat result = dualQuat::mul(dq1, dq2);
    
    // Store result: [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]
    out[idx * 8 + 0] = result.real.w;
    out[idx * 8 + 1] = result.real.x;
    out[idx * 8 + 2] = result.real.y;
    out[idx * 8 + 3] = result.real.z;
    out[idx * 8 + 4] = result.dual.w;
    out[idx * 8 + 5] = result.dual.x;
    out[idx * 8 + 6] = result.dual.y;
    out[idx * 8 + 7] = result.dual.z;
}

// Kernel: Dual quaternion conjugate
__global__ void ggml_dual_quaternion_conjugate_cuda(
    const float* dq,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Quaternion q_r(dq[idx * 8 + 0], dq[idx * 8 + 1], 
                  dq[idx * 8 + 2], dq[idx * 8 + 3]);
    Quaternion q_d(dq[idx * 8 + 4], dq[idx * 8 + 5], 
                  dq[idx * 8 + 6], dq[idx * 8 + 7]);
    dualQuat dq_obj(q_r, q_d);
    
    dualQuat result = dualQuat::conjugate(dq_obj);
    
    out[idx * 8 + 0] = result.real.w;
    out[idx * 8 + 1] = result.real.x;
    out[idx * 8 + 2] = result.real.y;
    out[idx * 8 + 3] = result.real.z;
    out[idx * 8 + 4] = result.dual.w;
    out[idx * 8 + 5] = result.dual.x;
    out[idx * 8 + 6] = result.dual.y;
    out[idx * 8 + 7] = result.dual.z;
}

// Kernel: Dual quaternion normalize
__global__ void ggml_dual_quaternion_normalize_cuda(
    const float* dq,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Quaternion q_r(dq[idx * 8 + 0], dq[idx * 8 + 1], 
                  dq[idx * 8 + 2], dq[idx * 8 + 3]);
    Quaternion q_d(dq[idx * 8 + 4], dq[idx * 8 + 5], 
                  dq[idx * 8 + 6], dq[idx * 8 + 7]);
    dualQuat dq_obj(q_r, q_d);
    
    dualQuat result = dualQuat::normalize(dq_obj);
    
    out[idx * 8 + 0] = result.real.w;
    out[idx * 8 + 1] = result.real.x;
    out[idx * 8 + 2] = result.real.y;
    out[idx * 8 + 3] = result.real.z;
    out[idx * 8 + 4] = result.dual.w;
    out[idx * 8 + 5] = result.dual.x;
    out[idx * 8 + 6] = result.dual.y;
    out[idx * 8 + 7] = result.dual.z;
}

// Host wrapper functions
extern "C" {
    void ggml_dual_quaternion_hamilton_product_cuda_launch(
        const float* q1,
        const float* q2,
        float* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_dual_quaternion_hamilton_product_cuda<<<blocks, threads, 0, stream>>>(q1, q2, out, n);
    }
    
    void ggml_dual_quaternion_conjugate_cuda_launch(
        const float* dq,
        float* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_dual_quaternion_conjugate_cuda<<<blocks, threads, 0, stream>>>(dq, out, n);
    }
    
    void ggml_dual_quaternion_normalize_cuda_launch(
        const float* dq,
        float* out,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_dual_quaternion_normalize_cuda<<<blocks, threads, 0, stream>>>(dq, out, n);
    }
}
