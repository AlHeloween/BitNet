#include <cuda_runtime.h>
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"
#include "../../lib/cuda_math.h"

using namespace DynaMap::math;

// CUDA kernel: Compute screw motion
__global__ void ggml_screw_compute_motion_cuda(
    const float* omega,  // [n, 3]
    const float* v,      // [n, 3]
    float* dq_out,       // [n, 8] dual quaternion
    const float* dt,     // [n] or scalar
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float dt_val = dt[0];  // Simplified - would handle broadcasting
    
    // Compute rotation quaternion from omega
    float omega_dt[3] = {
        omega[idx * 3 + 0] * dt_val,
        omega[idx * 3 + 1] * dt_val,
        omega[idx * 3 + 2] * dt_val,
    };
    
    float omega_norm = sqrtf(omega_dt[0] * omega_dt[0] + omega_dt[1] * omega_dt[1] + omega_dt[2] * omega_dt[2]);
    
    float half_angle = omega_norm / 2.0f;
    float cos_half = cosf(half_angle);
    float sin_half = sinf(half_angle);
    
    // Quaternion: [w, x, y, z]
    if (omega_norm > 1e-6f) {
        dq_out[idx * 8 + 0] = cos_half;  // w
        dq_out[idx * 8 + 1] = sin_half * omega_dt[0] / omega_norm;  // x
        dq_out[idx * 8 + 2] = sin_half * omega_dt[1] / omega_norm;  // y
        dq_out[idx * 8 + 3] = sin_half * omega_dt[2] / omega_norm;  // z
    } else {
        dq_out[idx * 8 + 0] = 1.0f;
        dq_out[idx * 8 + 1] = 0.0f;
        dq_out[idx * 8 + 2] = 0.0f;
        dq_out[idx * 8 + 3] = 0.0f;
    }
    
    // Dual part: translation
    float t[3] = {
        v[idx * 3 + 0] * dt_val,
        v[idx * 3 + 1] * dt_val,
        v[idx * 3 + 2] * dt_val,
    };
    
    dq_out[idx * 8 + 4] = 0.0f;  // dual w
    dq_out[idx * 8 + 5] = 0.5f * t[0];  // dual x
    dq_out[idx * 8 + 6] = 0.5f * t[1];  // dual y
    dq_out[idx * 8 + 7] = 0.5f * t[2];  // dual z
}

// Host wrapper
extern "C" {
    void ggml_screw_compute_motion_cuda_launch(
        const float* omega,
        const float* v,
        float* dq_out,
        const float* dt,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_screw_compute_motion_cuda<<<blocks, threads, 0, stream>>>(
            omega, v, dq_out, dt, n
        );
    }
}
