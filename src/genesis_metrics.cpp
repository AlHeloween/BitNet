#include "genesis_metrics.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

// Helper: Extract translation from dual quaternion
static void extract_translation_dual_quat(
    const float* dq,  // (8,)
    float* translation  // (3,)
) {
    // Real part (rotation quaternion)
    float real[4] = {dq[0], dq[1], dq[2], dq[3]};
    float real_norm = sqrtf(real[0]*real[0] + real[1]*real[1] + real[2]*real[2] + real[3]*real[3]);
    if (real_norm < 1e-8f) {
        translation[0] = 0.0f;
        translation[1] = 0.0f;
        translation[2] = 0.0f;
        return;
    }
    real[0] /= real_norm;
    real[1] /= real_norm;
    real[2] /= real_norm;
    real[3] /= real_norm;
    
    // Dual part
    float dual[4] = {dq[4], dq[5], dq[6], dq[7]};
    
    // Compute: dual * conjugate(real)
    float real_conj[4] = {real[0], -real[1], -real[2], -real[3]};
    float prod[4];
    prod[0] = dual[0]*real_conj[0] - dual[1]*real_conj[1] - dual[2]*real_conj[2] - dual[3]*real_conj[3];
    prod[1] = dual[0]*real_conj[1] + dual[1]*real_conj[0] + dual[2]*real_conj[3] - dual[3]*real_conj[2];
    prod[2] = dual[0]*real_conj[2] - dual[1]*real_conj[3] + dual[2]*real_conj[0] + dual[3]*real_conj[1];
    prod[3] = dual[0]*real_conj[3] + dual[1]*real_conj[2] - dual[2]*real_conj[1] + dual[3]*real_conj[0];
    
    // Extract vector part and multiply by 2
    translation[0] = 2.0f * prod[1];
    translation[1] = 2.0f * prod[2];
    translation[2] = 2.0f * prod[3];
}

// Helper: Compute geodesic distance between two dual quaternions
static float dual_quat_geodesic_distance(
    const float* dq1,  // (8,)
    const float* dq2,  // (8,)
    float w_rot,
    float w_trans
) {
    // Normalize real parts
    float real1[4] = {dq1[0], dq1[1], dq1[2], dq1[3]};
    float real2[4] = {dq2[0], dq2[1], dq2[2], dq2[3]};
    
    float norm1 = sqrtf(real1[0]*real1[0] + real1[1]*real1[1] + real1[2]*real1[2] + real1[3]*real1[3]);
    float norm2 = sqrtf(real2[0]*real2[0] + real2[1]*real2[1] + real2[2]*real2[2] + real2[3]*real2[3]);
    
    if (norm1 < 1e-8f || norm2 < 1e-8f) {
        return 1e6f;  // Large distance for invalid quaternions
    }
    
    real1[0] /= norm1; real1[1] /= norm1; real1[2] /= norm1; real1[3] /= norm1;
    real2[0] /= norm2; real2[1] /= norm2; real2[2] /= norm2; real2[3] /= norm2;
    
    // Rotation distance: geodesic on SO(3)
    // Compute R1^T * R2 = conjugate(R1) * R2
    float real1_conj[4] = {real1[0], -real1[1], -real1[2], -real1[3]};
    float rot_diff[4];
    rot_diff[0] = real1_conj[0]*real2[0] - real1_conj[1]*real2[1] - real1_conj[2]*real2[2] - real1_conj[3]*real2[3];
    rot_diff[1] = real1_conj[0]*real2[1] + real1_conj[1]*real2[0] + real1_conj[2]*real2[3] - real1_conj[3]*real2[2];
    rot_diff[2] = real1_conj[0]*real2[2] - real1_conj[1]*real2[3] + real1_conj[2]*real2[0] + real1_conj[3]*real2[1];
    rot_diff[3] = real1_conj[0]*real2[3] + real1_conj[1]*real2[2] - real1_conj[2]*real2[1] + real1_conj[3]*real2[0];
    
    // Angle = 2 * arccos(|dot(R1, R2)|)
    float dot_product = fabsf(rot_diff[0]);
    if (dot_product > 1.0f) dot_product = 1.0f;
    float rot_angle = 2.0f * acosf(dot_product);
    float rot_dist = rot_angle;
    
    // Translation distance: Euclidean
    float trans1[3], trans2[3];
    extract_translation_dual_quat(dq1, trans1);
    extract_translation_dual_quat(dq2, trans2);
    
    float trans_diff[3] = {trans1[0] - trans2[0], trans1[1] - trans2[1], trans1[2] - trans2[2]};
    float trans_dist = sqrtf(trans_diff[0]*trans_diff[0] + trans_diff[1]*trans_diff[1] + trans_diff[2]*trans_diff[2]);
    
    // Weighted combination
    return w_rot * rot_dist + w_trans * trans_dist;
}

// Check if dual quaternion is invalid (NaN or Inf)
static bool is_invalid_dq(const float* dq) {
    for (int i = 0; i < 8; i++) {
        if (isnan(dq[i]) || isinf(dq[i])) {
            return true;
        }
    }
    return false;
}

int genesis_compute_metrics(
    const float* buffer_embeddings,
    const float* medoids,
    const int32_t* labels,
    const float* previous_medoids,
    int32_t N,
    int32_t K,
    float w_rot,
    float w_trans,
    bool converged,
    int32_t iterations,
    genesis_metrics_t* metrics
) {
    if (!buffer_embeddings || !medoids || !labels || !metrics || N < 1 || K < 1) {
        return -1;
    }
    
    // Initialize metrics
    memset(metrics, 0, sizeof(genesis_metrics_t));
    metrics->k = K;
    metrics->converged = converged;
    metrics->iterations = iterations;
    
    // Allocate cluster sizes array
    metrics->cluster_sizes = (int32_t*)malloc(K * sizeof(int32_t));
    if (!metrics->cluster_sizes) {
        return -1;
    }
    memset(metrics->cluster_sizes, 0, K * sizeof(int32_t));
    
    // 1. Compute cost metrics (distances from points to their assigned medoids)
    std::vector<float> costs(N);
    float total_cost = 0.0f;
    float max_cost = 0.0f;
    
    for (int32_t i = 0; i < N; i++) {
        const float* point = buffer_embeddings + i * 8;
        int32_t medoid_idx = labels[i];
        if (medoid_idx < 0 || medoid_idx >= K) {
            free(metrics->cluster_sizes);
            return -1;
        }
        const float* medoid = medoids + medoid_idx * 8;
        
        float cost = dual_quat_geodesic_distance(point, medoid, w_rot, w_trans);
        costs[i] = cost;
        total_cost += cost;
        if (cost > max_cost) {
            max_cost = cost;
        }
        
        // Count cluster sizes
        metrics->cluster_sizes[medoid_idx]++;
    }
    
    metrics->total_cost = total_cost;
    metrics->mean_cost = (N > 0) ? (total_cost / N) : 0.0f;
    metrics->max_cost = max_cost;
    
    // 2. Compute medoid shift metrics (if previous medoids provided)
    if (previous_medoids != nullptr) {
        std::vector<float> shifts(K);
        float shift_sum = 0.0f;
        float shift_max = 0.0f;
        
        for (int32_t k = 0; k < K; k++) {
            const float* medoid = medoids + k * 8;
            const float* prev_medoid = previous_medoids + k * 8;
            float shift = dual_quat_geodesic_distance(medoid, prev_medoid, w_rot, w_trans);
            shifts[k] = shift;
            shift_sum += shift;
            if (shift > shift_max) {
                shift_max = shift;
            }
        }
        
        metrics->medoid_shift_mean = (K > 0) ? (shift_sum / K) : 0.0f;
        metrics->medoid_shift_max = shift_max;
        
        // Compute standard deviation
        float shift_var = 0.0f;
        for (int32_t k = 0; k < K; k++) {
            float diff = shifts[k] - metrics->medoid_shift_mean;
            shift_var += diff * diff;
        }
        metrics->medoid_shift_std = (K > 0) ? sqrtf(shift_var / K) : 0.0f;
    } else {
        metrics->medoid_shift_mean = 0.0f;
        metrics->medoid_shift_max = 0.0f;
        metrics->medoid_shift_std = 0.0f;
    }
    
    // 3. Compute cluster size metrics
    int32_t cluster_size_min = metrics->cluster_sizes[0];
    int32_t cluster_size_max = metrics->cluster_sizes[0];
    float cluster_size_sum = 0.0f;
    
    for (int32_t k = 0; k < K; k++) {
        int32_t size = metrics->cluster_sizes[k];
        cluster_size_sum += size;
        if (size < cluster_size_min) {
            cluster_size_min = size;
        }
        if (size > cluster_size_max) {
            cluster_size_max = size;
        }
    }
    
    metrics->cluster_size_mean = (K > 0) ? (cluster_size_sum / K) : 0.0f;
    metrics->cluster_size_min = cluster_size_min;
    metrics->cluster_size_max = cluster_size_max;
    
    // Compute standard deviation
    float cluster_size_var = 0.0f;
    for (int32_t k = 0; k < K; k++) {
        float diff = metrics->cluster_sizes[k] - metrics->cluster_size_mean;
        cluster_size_var += diff * diff;
    }
    metrics->cluster_size_std = (K > 0) ? sqrtf(cluster_size_var / K) : 0.0f;
    
    // 4. Compute invalid dual quaternion counts
    int32_t invalid_count = 0;
    for (int32_t i = 0; i < N; i++) {
        const float* dq = buffer_embeddings + i * 8;
        if (is_invalid_dq(dq)) {
            invalid_count++;
        }
    }
    
    metrics->invalid_dq_count = invalid_count;
    metrics->invalid_dq_ratio = (N > 0) ? (float(invalid_count) / N) : 0.0f;
    
    return 0;
}

void genesis_metrics_free(genesis_metrics_t* metrics) {
    if (metrics && metrics->cluster_sizes) {
        free(metrics->cluster_sizes);
        metrics->cluster_sizes = nullptr;
        metrics->k = 0;
    }
}
