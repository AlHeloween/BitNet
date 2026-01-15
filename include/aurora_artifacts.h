#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Artifact metadata structure (matches Python build_run_metadata)
typedef struct {
    char* run_id;              // Unique run identifier (UUID hex)
    char* timestamp_utc;       // UTC timestamp in ISO format
    char* inputs_hash;         // SHA256 hash of inputs
    char* git_rev;             // Git revision (short SHA) - NULL if unavailable
    bool git_dirty;            // Whether working directory is dirty
    char* git_error;           // Error message if git info failed - NULL if OK
    int32_t seed;              // Random seed used (-1 if not set)
    char** artifact_paths;     // Array of artifact file paths
    int artifact_count;        // Number of artifacts
} aurora_run_metadata_t;

// Initialize run metadata
// Returns NULL on failure
aurora_run_metadata_t* aurora_run_metadata_create(
    const char* inputs_json,   // JSON string of inputs
    int32_t seed,              // Random seed (-1 to skip)
    const char** artifact_paths,  // Array of artifact paths (NULL-terminated)
    int artifact_count
);

// Free run metadata
void aurora_run_metadata_free(aurora_run_metadata_t* metadata);

// Get current UTC timestamp as ISO string
// Caller must free the returned string
char* aurora_utc_timestamp_iso(void);

// Get git revision (short SHA)
// Returns NULL on failure, caller must free
char* aurora_git_rev_short(void);

// Check if git working directory is dirty
bool aurora_git_is_dirty(void);

// Generate UUID hex string for run_id
// Caller must free the returned string
char* aurora_generate_run_id(void);

// Hash JSON string (SHA256)
// Caller must free the returned string
char* aurora_hash_json(const char* json_str);

// Set CUDA random seed for determinism
void aurora_cuda_set_seed(int32_t seed);

// Enable CUDA deterministic operations
// Note: May impact performance
void aurora_cuda_set_deterministic(bool enable);

#ifdef __cplusplus
}
#endif
