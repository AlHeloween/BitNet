#include "ggml-dual-complex.h"
#include "ggml.h"
#include "../../../lib/math_dualPhaser.h"

#include <cmath>
#include <cstring>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
// CUDA kernel wrappers will be called from ggml-backend-cuda
// For now, we use CPU implementation
#endif

using namespace DynaMap::math;

// Helper: Convert GGML tensor to dualPhaser array
static void tensor_to_dualphaser(
    const struct ggml_tensor* tensor,
    dualPhaser* out,
    int n_elements
) {
    const float* data = (const float*)tensor->data;
    for (int i = 0; i < n_elements; i++) {
        float2 real = make_float2(data[i * 2], data[i * 2 + 1]);
        float2 dual = make_float2(0.0f, 0.0f);  // Default dual to zero if not provided
        out[i] = dualPhaser(real, dual);
    }
}

// Helper: Convert dualPhaser array to GGML tensor
static void dualphaser_to_tensor(
    const dualPhaser* in,
    struct ggml_tensor* tensor,
    int n_elements
) {
    float* data = (float*)tensor->data;
    for (int i = 0; i < n_elements; i++) {
        data[i * 2] = in[i].real.x;
        data[i * 2 + 1] = in[i].real.y;
        // Dual component would go in [i * 2 + 2] if we had 4 floats per element
        // For now, we use [..., 2] format: [primal, dual] per element
    }
}

struct ggml_tensor* ggml_dual_complex(
    struct ggml_context* ctx,
    struct ggml_tensor* primal,
    struct ggml_tensor* dual
) {
    // Stack primal and dual along last dimension
    // primal: [..., dim], dual: [..., dim] -> output: [..., dim, 2]
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 
        ggml_n_dims(primal) + 1, primal->ne);
    result->ne[ggml_n_dims(primal)] = 2;
    
    // Copy data: [..., dim, 0] = primal, [..., dim, 1] = dual
    float* out_data = (float*)result->data;
    const float* primal_data = (const float*)primal->data;
    const float* dual_data = (const float*)dual->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(primal); i++) {
        n_elements *= primal->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        out_data[i * 2] = primal_data[i];
        out_data[i * 2 + 1] = dual_data[i];
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_complex_primal(
    struct ggml_context* ctx,
    struct ggml_tensor* dc
) {
    // Extract [..., 0] from [..., 2]
    int n_dims = ggml_n_dims(dc) - 1;
    int64_t* ne = new int64_t[n_dims];
    for (int i = 0; i < n_dims; i++) {
        ne[i] = dc->ne[i];
    }
    
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    delete[] ne;
    
    const float* dc_data = (const float*)dc->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        n_elements *= result->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        out_data[i] = dc_data[i * 2];
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_complex_dual(
    struct ggml_context* ctx,
    struct ggml_tensor* dc
) {
    // Extract [..., 1] from [..., 2]
    int n_dims = ggml_n_dims(dc) - 1;
    int64_t* ne = new int64_t[n_dims];
    for (int i = 0; i < n_dims; i++) {
        ne[i] = dc->ne[i];
    }
    
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    delete[] ne;
    
    const float* dc_data = (const float*)dc->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        n_elements *= result->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        out_data[i] = dc_data[i * 2 + 1];
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_complex_add(
    struct ggml_context* ctx,
    struct ggml_tensor* dc1,
    struct ggml_tensor* dc2
) {
    GGML_ASSERT(ggml_n_dims(dc1) == ggml_n_dims(dc2));
    for (int i = 0; i < ggml_n_dims(dc1); i++) {
        GGML_ASSERT(dc1->ne[i] == dc2->ne[i]);
    }
    
    struct ggml_tensor* result = ggml_dup_tensor(ctx, dc1);
    
    const float* dc1_data = (const float*)dc1->data;
    const float* dc2_data = (const float*)dc2->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(dc1) - 1; i++) {
        n_elements *= dc1->ne[i];
    }
    
    // Process each dual-complex element
    for (int i = 0; i < n_elements; i++) {
        float4 packed1 = make_float4(dc1_data[i * 2], dc1_data[i * 2 + 1], 0.0f, 0.0f);
        float4 packed2 = make_float4(dc2_data[i * 2], dc2_data[i * 2 + 1], 0.0f, 0.0f);
        
        dualPhaser dp1 = dualphaser_unpack(packed1);
        dualPhaser dp2 = dualphaser_unpack(packed2);
        
        dualPhaser result_dp = dualphaser_add(dp1, dp2);
        float4 result_packed = dualphaser_pack(result_dp);
        
        out_data[i * 2] = result_packed.x;
        out_data[i * 2 + 1] = result_packed.y;
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_complex_mul(
    struct ggml_context* ctx,
    struct ggml_tensor* dc1,
    struct ggml_tensor* dc2
) {
    GGML_ASSERT(ggml_n_dims(dc1) == ggml_n_dims(dc2));
    for (int i = 0; i < ggml_n_dims(dc1); i++) {
        GGML_ASSERT(dc1->ne[i] == dc2->ne[i]);
    }
    
    struct ggml_tensor* result = ggml_dup_tensor(ctx, dc1);
    
    const float* dc1_data = (const float*)dc1->data;
    const float* dc2_data = (const float*)dc2->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(dc1) - 1; i++) {
        n_elements *= dc1->ne[i];
    }
    
    // Process each dual-complex element using existing dualphaser_mul
    for (int i = 0; i < n_elements; i++) {
        // Pack as float4: [real.x, real.y, dual.x, dual.y]
        // For [..., 2] format, we only have primal, so dual defaults to zero
        float4 packed1 = make_float4(dc1_data[i * 2], dc1_data[i * 2 + 1], 0.0f, 0.0f);
        float4 packed2 = make_float4(dc2_data[i * 2], dc2_data[i * 2 + 1], 0.0f, 0.0f);
        
        dualPhaser dp1 = dualphaser_unpack(packed1);
        dualPhaser dp2 = dualphaser_unpack(packed2);
        
        dualPhaser result_dp = dualphaser_mul(dp1, dp2);
        float4 result_packed = dualphaser_pack(result_dp);
        
        out_data[i * 2] = result_packed.x;
        out_data[i * 2 + 1] = result_packed.y;
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_complex_conjugate(
    struct ggml_context* ctx,
    struct ggml_tensor* dc
) {
    struct ggml_tensor* result = ggml_dup_tensor(ctx, dc);
    
    const float* dc_data = (const float*)dc->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(dc) - 1; i++) {
        n_elements *= dc->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        float4 packed = make_float4(dc_data[i * 2], dc_data[i * 2 + 1], 0.0f, 0.0f);
        dualPhaser dp = dualphaser_unpack(packed);
        dualPhaser result_dp = dualphaser_conjugate(dp);
        float4 result_packed = dualphaser_pack(result_dp);
        
        out_data[i * 2] = result_packed.x;
        out_data[i * 2 + 1] = result_packed.y;
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_complex_normalize(
    struct ggml_context* ctx,
    struct ggml_tensor* dc
) {
    struct ggml_tensor* result = ggml_dup_tensor(ctx, dc);
    
    const float* dc_data = (const float*)dc->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(dc) - 1; i++) {
        n_elements *= dc->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        float4 packed = make_float4(dc_data[i * 2], dc_data[i * 2 + 1], 0.0f, 0.0f);
        dualPhaser dp = dualphaser_unpack(packed);
        dualPhaser result_dp = dualphaser_normalize(dp);
        float4 result_packed = dualphaser_pack(result_dp);
        
        out_data[i * 2] = result_packed.x;
        out_data[i * 2 + 1] = result_packed.y;
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_complex_softmax(
    struct ggml_context* ctx,
    struct ggml_tensor* dc
) {
    // Dual-softmax: softmax(primal) for primal, softmax(primal) * (dual - mean(dual)) for dual
    // For now, use standard softmax on primal, keep dual as is
    struct ggml_tensor* primal = ggml_dual_complex_primal(ctx, dc);
    struct ggml_tensor* softmax_primal = ggml_soft_max(ctx, primal);
    
    // Reconstruct dual-complex with softmax primal
    struct ggml_tensor* dual = ggml_dual_complex_dual(ctx, dc);
    return ggml_dual_complex(ctx, softmax_primal, dual);
}
