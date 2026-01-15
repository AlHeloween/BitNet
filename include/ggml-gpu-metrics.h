#ifndef GGML_GPU_METRICS_H
#define GGML_GPU_METRICS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU metrics collection utilities for CUDA operations
// Stage 6: Real model tests with GPU metrics

// CUDA event timing helper
// Returns elapsed time in milliseconds
float ggml_cuda_time_operation(void (*operation)(void*), void* user_data);

// Get GPU memory usage
size_t ggml_cuda_get_memory_allocated();
size_t ggml_cuda_get_memory_reserved();

// Log GPU metrics (for extraction by Python)
void ggml_cuda_log_metrics(
    const char* operation_name,
    float elapsed_ms,
    size_t memory_bytes
);

#ifdef __cplusplus
}
#endif

#endif // GGML_GPU_METRICS_H
