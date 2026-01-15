#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// Aurora configuration structure
// Mirrors settings.json structure for C++ compatibility
struct aurora_config {
    // Paths
    char bitnet_gguf[512];
    char bitnet_dir[512];
    char llama_cli[512];
    char text_model[512];
    char deepseek_vl_model[512];
    char wiki_dataset_dir[512];
    
    // Navigation paradigm settings
    bool navigation_enabled;
    float nav_tau;
    int32_t nav_max_depth;
    int32_t nav_max_nodes;
    int32_t nav_window_size;
    int32_t nav_k_read;
    
    // GPU metrics
    bool gpu_metrics_enabled;
    
    // Build settings
    char nvcc_ccbin[256];
    char nvptx_toolchain[256];
};

// Initialize config with defaults
void aurora_config_init_defaults(struct aurora_config* config);

// Load config from environment variables
// Reads AURORA_* environment variables and populates config
void aurora_config_load_from_env(struct aurora_config* config);

// Load config from JSON file (simple parser)
// Returns 0 on success, -1 on error
int aurora_config_load_from_json(struct aurora_config* config, const char* json_path);

// Save config to JSON file
// Returns 0 on success, -1 on error
int aurora_config_save_to_json(const struct aurora_config* config, const char* json_path);

// Get config value by key (for compatibility with Python settings)
// Returns pointer to value string, or NULL if not found
const char* aurora_config_get(const struct aurora_config* config, const char* key);

// Set config value by key
// Returns 0 on success, -1 on error (invalid key)
int aurora_config_set(struct aurora_config* config, const char* key, const char* value);

#ifdef __cplusplus
}
#endif
