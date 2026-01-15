#include "ggml-gpu-metrics.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#endif

float ggml_cuda_time_operation(void (*operation)(void*), void* user_data) {
#ifdef GGML_USE_CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    if (operation) {
        operation(user_data);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
#else
    (void)operation;
    (void)user_data;
    return 0.0f;
#endif
}

size_t ggml_cuda_get_memory_allocated() {
#ifdef GGML_USE_CUDA
    size_t allocated = 0;
    cudaMemGetInfo(NULL, &allocated);
    // Get total allocated (not free)
    size_t free = 0;
    cudaMemGetInfo(&free, &allocated);
    return allocated - free;
#else
    return 0;
#endif
}

size_t ggml_cuda_get_memory_reserved() {
#ifdef GGML_USE_CUDA
    size_t reserved = 0;
    cudaMemGetInfo(NULL, &reserved);
    return reserved;
#else
    return 0;
#endif
}

void ggml_cuda_log_metrics(
    const char* operation_name,
    float elapsed_ms,
    size_t memory_bytes
) {
    // Log in format parseable by Python: AURORA_GPU_<operation>=<value>
    if (operation_name) {
        printf("AURORA_GPU_%s_TIME=%.3f\n", operation_name, elapsed_ms);
        printf("AURORA_GPU_%s_MEMORY=%zu\n", operation_name, memory_bytes);
        fflush(stdout);
    }
}
