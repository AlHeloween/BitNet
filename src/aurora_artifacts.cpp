#include "aurora_artifacts.h"

#include <cstring>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <random>

#ifdef __cplusplus
extern "C" {
#endif

// Simple SHA256 implementation (no external dependencies)
// Using a simplified hash for inputs_hash (can be replaced with proper SHA256 if needed)

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#endif

// Generate UUID v4 hex string (32 hex chars, no dashes)
static char* generate_uuid_hex() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    char* uuid = (char*)malloc(33);  // 32 hex chars + null terminator
    if (!uuid) return NULL;
    
    for (int i = 0; i < 32; i++) {
        int val = dis(gen);
        uuid[i] = (val < 10) ? ('0' + val) : ('a' + val - 10);
    }
    uuid[32] = '\0';
    return uuid;
}

// Get current UTC timestamp as ISO string (YYYY-MM-DDTHH:MM:SSZ)
char* aurora_utc_timestamp_iso(void) {
    time_t now;
    time(&now);
    struct tm* utc = gmtime(&now);
    
    char* timestamp = (char*)malloc(21);  // 20 chars + null terminator
    if (!timestamp) return NULL;
    
    snprintf(timestamp, 21, "%04d-%02d-%02dT%02d:%02d:%02dZ",
             utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
             utc->tm_hour, utc->tm_min, utc->tm_sec);
    return timestamp;
}

// Get git revision (short SHA) - simplified version
// In production, would use libgit2 or exec git command
char* aurora_git_rev_short(void) {
    // For now, return NULL (would need git integration)
    // TODO: Implement using libgit2 or exec "git rev-parse --short HEAD"
    return NULL;
}

// Check if git working directory is dirty
bool aurora_git_is_dirty(void) {
    // For now, return false (would need git integration)
    // TODO: Implement using libgit2 or exec "git status --porcelain"
    return false;
}

// Generate run ID (UUID hex)
char* aurora_generate_run_id(void) {
    return generate_uuid_hex();
}

// Simple hash function (FNV-1a variant) for inputs_hash
// In production, replace with proper SHA256 if available
static uint64_t fnv1a_hash(const char* str, size_t len) {
    const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)(unsigned char)str[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

// Hash JSON string (simplified - uses FNV-1a, produces 16 hex chars)
// For full SHA256 compatibility, integrate a proper SHA256 library
char* aurora_hash_json(const char* json_str) {
    if (!json_str) return NULL;
    
    size_t len = strlen(json_str);
    uint64_t hash = fnv1a_hash(json_str, len);
    
    // Convert to hex string (16 hex chars for 64-bit hash)
    char* hex = (char*)malloc(17);  // 16 hex chars + null terminator
    if (!hex) return NULL;
    
    snprintf(hex, 17, "%016llx", (unsigned long long)hash);
    return hex;
}

// Set CUDA random seed
void aurora_cuda_set_seed(int32_t seed) {
#ifdef GGML_USE_CUDA
    if (seed < 0) return;
    
    // Set CUDA device seeds
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; i++) {
        cudaSetDevice(i);
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)seed);
        curandDestroyGenerator(gen);
    }
    
    // Reset to default device
    cudaSetDevice(0);
#endif
}

// Enable CUDA deterministic operations
void aurora_cuda_set_deterministic(bool enable) {
#ifdef GGML_USE_CUDA
    // Note: This is a placeholder - actual implementation would set
    // CUBLAS and cuDNN deterministic flags if those libraries are used
    // For GGML, determinism is handled at the kernel level
    (void)enable;  // Suppress unused parameter warning
#endif
}

// Create run metadata
aurora_run_metadata_t* aurora_run_metadata_create(
    const char* inputs_json,
    int32_t seed,
    const char** artifact_paths,
    int artifact_count
) {
    aurora_run_metadata_t* meta = (aurora_run_metadata_t*)malloc(sizeof(aurora_run_metadata_t));
    if (!meta) return NULL;
    
    memset(meta, 0, sizeof(aurora_run_metadata_t));
    
    // Generate run_id
    meta->run_id = aurora_generate_run_id();
    if (!meta->run_id) {
        free(meta);
        return NULL;
    }
    
    // Get timestamp
    meta->timestamp_utc = aurora_utc_timestamp_iso();
    if (!meta->timestamp_utc) {
        free(meta->run_id);
        free(meta);
        return NULL;
    }
    
    // Hash inputs
    if (inputs_json) {
        meta->inputs_hash = aurora_hash_json(inputs_json);
        if (!meta->inputs_hash) {
            free(meta->run_id);
            free(meta->timestamp_utc);
            free(meta);
            return NULL;
        }
    } else {
        meta->inputs_hash = NULL;
    }
    
    // Get git info
    meta->git_rev = aurora_git_rev_short();
    meta->git_dirty = aurora_git_is_dirty();
    meta->git_error = NULL;  // TODO: Capture git errors if needed
    
    // Set seed
    meta->seed = seed;
    
    // Copy artifact paths
    if (artifact_paths && artifact_count > 0) {
        meta->artifact_paths = (char**)malloc(artifact_count * sizeof(char*));
        if (!meta->artifact_paths) {
            aurora_run_metadata_free(meta);
            return NULL;
        }
        meta->artifact_count = artifact_count;
        for (int i = 0; i < artifact_count; i++) {
            if (artifact_paths[i]) {
                meta->artifact_paths[i] = strdup(artifact_paths[i]);
            } else {
                meta->artifact_paths[i] = NULL;
            }
        }
    } else {
        meta->artifact_paths = NULL;
        meta->artifact_count = 0;
    }
    
    return meta;
}

// Free run metadata
void aurora_run_metadata_free(aurora_run_metadata_t* metadata) {
    if (!metadata) return;
    
    if (metadata->run_id) free(metadata->run_id);
    if (metadata->timestamp_utc) free(metadata->timestamp_utc);
    if (metadata->inputs_hash) free(metadata->inputs_hash);
    if (metadata->git_rev) free(metadata->git_rev);
    if (metadata->git_error) free(metadata->git_error);
    
    if (metadata->artifact_paths) {
        for (int i = 0; i < metadata->artifact_count; i++) {
            if (metadata->artifact_paths[i]) {
                free(metadata->artifact_paths[i]);
            }
        }
        free(metadata->artifact_paths);
    }
    
    free(metadata);
}

#ifdef __cplusplus
}
#endif
