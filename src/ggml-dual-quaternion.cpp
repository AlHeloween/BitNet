#include "ggml-dual-quaternion.h"
#include "ggml.h"
#include "../../../lib/math_dualQuat.h"
#include "../../../lib/math_Quat.h"

#include <cmath>
#include <cstring>

using namespace DynaMap::math;

struct ggml_tensor* ggml_dual_quaternion_hamilton_product(
    struct ggml_context* ctx,
    struct ggml_tensor* q1,
    struct ggml_tensor* q2
) {
    GGML_ASSERT(ggml_n_dims(q1) == ggml_n_dims(q2));
    for (int i = 0; i < ggml_n_dims(q1); i++) {
        GGML_ASSERT(q1->ne[i] == q2->ne[i]);
    }
    GGML_ASSERT(q1->ne[ggml_n_dims(q1) - 1] == 8);  // Last dim must be 8
    
    struct ggml_tensor* result = ggml_dup_tensor(ctx, q1);
    
    const float* q1_data = (const float*)q1->data;
    const float* q2_data = (const float*)q2->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(q1) - 1; i++) {
        n_elements *= q1->ne[i];
    }
    
    // Process each dual quaternion using existing dualQuat::mul()
    for (int i = 0; i < n_elements; i++) {
        // Load quaternions: [w, x, y, z]
        Quaternion q1_r(q1_data[i * 8 + 0], q1_data[i * 8 + 1], 
                       q1_data[i * 8 + 2], q1_data[i * 8 + 3]);
        Quaternion q1_d(q1_data[i * 8 + 4], q1_data[i * 8 + 5], 
                       q1_data[i * 8 + 6], q1_data[i * 8 + 7]);
        dualQuat dq1(q1_r, q1_d);
        
        Quaternion q2_r(q2_data[i * 8 + 0], q2_data[i * 8 + 1], 
                       q2_data[i * 8 + 2], q2_data[i * 8 + 3]);
        Quaternion q2_d(q2_data[i * 8 + 4], q2_data[i * 8 + 5], 
                       q2_data[i * 8 + 6], q2_data[i * 8 + 7]);
        dualQuat dq2(q2_r, q2_d);
        
        // Use existing Hamilton product
        dualQuat result_dq = dualQuat::mul(dq1, dq2);
        
        // Store result: [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]
        out_data[i * 8 + 0] = result_dq.real.w;
        out_data[i * 8 + 1] = result_dq.real.x;
        out_data[i * 8 + 2] = result_dq.real.y;
        out_data[i * 8 + 3] = result_dq.real.z;
        out_data[i * 8 + 4] = result_dq.dual.w;
        out_data[i * 8 + 5] = result_dq.dual.x;
        out_data[i * 8 + 6] = result_dq.dual.y;
        out_data[i * 8 + 7] = result_dq.dual.z;
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_quaternion_conjugate(
    struct ggml_context* ctx,
    struct ggml_tensor* dq
) {
    struct ggml_tensor* result = ggml_dup_tensor(ctx, dq);
    
    const float* dq_data = (const float*)dq->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(dq) - 1; i++) {
        n_elements *= dq->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        Quaternion q_r(dq_data[i * 8 + 0], dq_data[i * 8 + 1], 
                      dq_data[i * 8 + 2], dq_data[i * 8 + 3]);
        Quaternion q_d(dq_data[i * 8 + 4], dq_data[i * 8 + 5], 
                      dq_data[i * 8 + 6], dq_data[i * 8 + 7]);
        dualQuat dq(q_r, q_d);
        
        dualQuat result_dq = dualQuat::conjugate(dq);
        
        out_data[i * 8 + 0] = result_dq.real.w;
        out_data[i * 8 + 1] = result_dq.real.x;
        out_data[i * 8 + 2] = result_dq.real.y;
        out_data[i * 8 + 3] = result_dq.real.z;
        out_data[i * 8 + 4] = result_dq.dual.w;
        out_data[i * 8 + 5] = result_dq.dual.x;
        out_data[i * 8 + 6] = result_dq.dual.y;
        out_data[i * 8 + 7] = result_dq.dual.z;
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_quaternion_normalize(
    struct ggml_context* ctx,
    struct ggml_tensor* dq
) {
    struct ggml_tensor* result = ggml_dup_tensor(ctx, dq);
    
    const float* dq_data = (const float*)dq->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(dq) - 1; i++) {
        n_elements *= dq->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        Quaternion q_r(dq_data[i * 8 + 0], dq_data[i * 8 + 1], 
                      dq_data[i * 8 + 2], dq_data[i * 8 + 3]);
        Quaternion q_d(dq_data[i * 8 + 4], dq_data[i * 8 + 5], 
                      dq_data[i * 8 + 6], dq_data[i * 8 + 7]);
        dualQuat dq_obj(q_r, q_d);
        
        dualQuat result_dq = dualQuat::normalize(dq_obj);
        
        out_data[i * 8 + 0] = result_dq.real.w;
        out_data[i * 8 + 1] = result_dq.real.x;
        out_data[i * 8 + 2] = result_dq.real.y;
        out_data[i * 8 + 3] = result_dq.real.z;
        out_data[i * 8 + 4] = result_dq.dual.w;
        out_data[i * 8 + 5] = result_dq.dual.x;
        out_data[i * 8 + 6] = result_dq.dual.y;
        out_data[i * 8 + 7] = result_dq.dual.z;
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_quaternion_inverse(
    struct ggml_context* ctx,
    struct ggml_tensor* dq
) {
    struct ggml_tensor* result = ggml_dup_tensor(ctx, dq);
    
    const float* dq_data = (const float*)dq->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < ggml_n_dims(dq) - 1; i++) {
        n_elements *= dq->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        Quaternion q_r(dq_data[i * 8 + 0], dq_data[i * 8 + 1], 
                      dq_data[i * 8 + 2], dq_data[i * 8 + 3]);
        Quaternion q_d(dq_data[i * 8 + 4], dq_data[i * 8 + 5], 
                      dq_data[i * 8 + 6], dq_data[i * 8 + 7]);
        dualQuat dq_obj(q_r, q_d);
        
        dualQuat result_dq = dq_obj.inverse();
        
        out_data[i * 8 + 0] = result_dq.real.w;
        out_data[i * 8 + 1] = result_dq.real.x;
        out_data[i * 8 + 2] = result_dq.real.y;
        out_data[i * 8 + 3] = result_dq.real.z;
        out_data[i * 8 + 4] = result_dq.dual.w;
        out_data[i * 8 + 5] = result_dq.dual.x;
        out_data[i * 8 + 6] = result_dq.dual.y;
        out_data[i * 8 + 7] = result_dq.dual.z;
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_quaternion_rotation(
    struct ggml_context* ctx,
    struct ggml_tensor* dq
) {
    // Extract [..., 0:4] from [..., 8]
    int n_dims = ggml_n_dims(dq);
    int64_t* ne = new int64_t[n_dims];
    for (int i = 0; i < n_dims - 1; i++) {
        ne[i] = dq->ne[i];
    }
    ne[n_dims - 1] = 4;
    
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    delete[] ne;
    
    const float* dq_data = (const float*)dq->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < n_dims - 1; i++) {
        n_elements *= result->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        out_data[i * 4 + 0] = dq_data[i * 8 + 0];  // w
        out_data[i * 4 + 1] = dq_data[i * 8 + 1];  // x
        out_data[i * 4 + 2] = dq_data[i * 8 + 2];  // y
        out_data[i * 4 + 3] = dq_data[i * 8 + 3];  // z
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_quaternion_translation(
    struct ggml_context* ctx,
    struct ggml_tensor* dq
) {
    // Extract [..., 4:8] from [..., 8]
    int n_dims = ggml_n_dims(dq);
    int64_t* ne = new int64_t[n_dims];
    for (int i = 0; i < n_dims - 1; i++) {
        ne[i] = dq->ne[i];
    }
    ne[n_dims - 1] = 4;
    
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    delete[] ne;
    
    const float* dq_data = (const float*)dq->data;
    float* out_data = (float*)result->data;
    
    int n_elements = 1;
    for (int i = 0; i < n_dims - 1; i++) {
        n_elements *= result->ne[i];
    }
    
    for (int i = 0; i < n_elements; i++) {
        out_data[i * 4 + 0] = dq_data[i * 8 + 4];  // w
        out_data[i * 4 + 1] = dq_data[i * 8 + 5];  // x
        out_data[i * 4 + 2] = dq_data[i * 8 + 6];  // y
        out_data[i * 4 + 3] = dq_data[i * 8 + 7];  // z
    }
    
    return result;
}

struct ggml_tensor* ggml_dual_quaternion_add(
    struct ggml_context* ctx,
    struct ggml_tensor* dq1,
    struct ggml_tensor* dq2
) {
    GGML_ASSERT(ggml_n_dims(dq1) == ggml_n_dims(dq2));
    for (int i = 0; i < ggml_n_dims(dq1); i++) {
        GGML_ASSERT(dq1->ne[i] == dq2->ne[i]);
    }
    
    // Use component-wise addition
    return ggml_add(ctx, dq1, dq2);
}

struct ggml_tensor* ggml_dual_quaternion_scale(
    struct ggml_context* ctx,
    struct ggml_tensor* dq,
    float scalar
) {
    return ggml_scale(ctx, dq, scalar);
}
