#ifndef GENESIS_METRICS_H
#define GENESIS_METRICS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Genesis acceptance metrics for SE(3) consolidation (Stage 5).
 * 
 * Provides metrics to evaluate Genesis step quality and stability:
 * - Finite cost (total clustering cost)
 * - Medoid shift magnitude (distance moved by medoids)
 * - Cluster sizes (distribution of cluster assignments)
 * - Invalid dual quaternion counts (NaN/Inf detection)
 */

typedef struct {
    // Cost metrics
    float total_cost;      // Total clustering cost (sum of distances to medoids)
    float mean_cost;       // Mean cost per point
    float max_cost;         // Maximum cost (worst point-to-medoid distance)
    
    // Medoid shift metrics
    float medoid_shift_mean;  // Mean shift magnitude across all medoids
    float medoid_shift_max;  // Maximum shift magnitude
    float medoid_shift_std;  // Standard deviation of shift magnitudes
    
    // Cluster size metrics
    int32_t* cluster_sizes;  // Size of each cluster (array of length k)
    int32_t k;               // Number of clusters
    float cluster_size_mean; // Mean cluster size
    float cluster_size_std;  // Standard deviation of cluster sizes
    int32_t cluster_size_min; // Minimum cluster size
    int32_t cluster_size_max; // Maximum cluster size
    
    // Invalid dual quaternion metrics
    int32_t invalid_dq_count;  // Number of invalid DQs (NaN/Inf)
    float invalid_dq_ratio;    // Ratio of invalid DQs to total
    
    // Convergence metrics
    bool converged;        // Whether clustering converged
    int32_t iterations;    // Number of iterations until convergence
} genesis_metrics_t;

/**
 * Compute Genesis acceptance metrics from clustering results.
 * 
 * @param buffer_embeddings Input dual quaternion embeddings (N, 8)
 * @param medoids Output medoids from clustering (K, 8)
 * @param labels Cluster assignments for each embedding (N,)
 * @param previous_medoids Previous medoids for shift calculation (K, 8) or NULL
 * @param N Number of embeddings
 * @param K Number of medoids
 * @param w_rot Weight for rotational distance component
 * @param w_trans Weight for translational distance component
 * @param converged Whether clustering converged
 * @param iterations Number of iterations until convergence
 * @param metrics Output metrics structure (must be pre-allocated)
 * 
 * @return 0 on success, non-zero on error
 */
int genesis_compute_metrics(
    const float* buffer_embeddings,  // (N, 8)
    const float* medoids,            // (K, 8)
    const int32_t* labels,           // (N,)
    const float* previous_medoids,  // (K, 8) or NULL
    int32_t N,
    int32_t K,
    float w_rot,
    float w_trans,
    bool converged,
    int32_t iterations,
    genesis_metrics_t* metrics
);

/**
 * Free resources allocated by genesis_compute_metrics.
 * 
 * @param metrics Metrics structure to free
 */
void genesis_metrics_free(genesis_metrics_t* metrics);

#ifdef __cplusplus
}
#endif

#endif // GENESIS_METRICS_H
