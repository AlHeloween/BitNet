#include "aurora_config.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

// Simple JSON parser helpers (minimal implementation)
static void skip_whitespace(const char** p) {
    while (**p && isspace(**p)) (*p)++;
}

static bool parse_string(const char** p, char* out, size_t out_size) {
    skip_whitespace(p);
    if (**p != '"') return false;
    (*p)++;
    
    size_t i = 0;
    while (**p && **p != '"' && i < out_size - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') { out[i++] = '\n'; (*p)++; }
            else if (**p == 't') { out[i++] = '\t'; (*p)++; }
            else if (**p == '\\') { out[i++] = '\\'; (*p)++; }
            else if (**p == '"') { out[i++] = '"'; (*p)++; }
            else { out[i++] = **p; (*p)++; }
        } else {
            out[i++] = **p;
            (*p)++;
        }
    }
    out[i] = '\0';
    
    if (**p == '"') (*p)++;
    return true;
}

static bool parse_number(const char** p, float* out_float, int32_t* out_int) {
    skip_whitespace(p);
    char* end;
    if (out_float) {
        *out_float = (float)strtod(*p, &end);
        if (end == *p) return false;
        *p = end;
    } else if (out_int) {
        *out_int = (int32_t)strtol(*p, &end, 10);
        if (end == *p) return false;
        *p = end;
    }
    return true;
}

static bool parse_bool(const char** p, bool* out) {
    skip_whitespace(p);
    if (strncmp(*p, "true", 4) == 0) {
        *out = true;
        *p += 4;
        return true;
    } else if (strncmp(*p, "false", 5) == 0) {
        *out = false;
        *p += 5;
        return true;
    }
    return false;
}

static bool find_key(const char** p, const char* key) {
    skip_whitespace(p);
    if (**p != '"') return false;
    (*p)++;
    size_t key_len = strlen(key);
    if (strncmp(*p, key, key_len) != 0) return false;
    *p += key_len;
    if (**p != '"') return false;
    (*p)++;
    skip_whitespace(p);
    if (**p != ':') return false;
    (*p)++;
    return true;
}

void aurora_config_init_defaults(struct aurora_config* config) {
    if (!config) return;
    
    memset(config, 0, sizeof(struct aurora_config));
    
    // Default paths (empty)
    config->bitnet_gguf[0] = '\0';
    config->bitnet_dir[0] = '\0';
    config->llama_cli[0] = '\0';
    config->text_model[0] = '\0';
    strncpy(config->deepseek_vl_model, "deepseek-ai/deepseek-vl-1.3b-base", sizeof(config->deepseek_vl_model) - 1);
    config->wiki_dataset_dir[0] = '\0';
    
    // Navigation defaults (from settings.json)
    config->navigation_enabled = true;
    config->nav_tau = 0.1f;
    config->nav_max_depth = 7;
    config->nav_max_nodes = 4;
    config->nav_window_size = 2048;
    config->nav_k_read = 8;
    
    // GPU metrics
    config->gpu_metrics_enabled = true;
    
    // Build settings
    config->nvcc_ccbin[0] = '\0';
    strncpy(config->nvptx_toolchain, "nightly-2024-09-05", sizeof(config->nvptx_toolchain) - 1);
}

void aurora_config_load_from_env(struct aurora_config* config) {
    if (!config) return;
    
    // Initialize defaults first
    aurora_config_init_defaults(config);
    
    // Load from environment variables
    const char* env_val;
    
    // Paths
    if ((env_val = getenv("AURORA_BITNET_GGUF"))) {
        strncpy(config->bitnet_gguf, env_val, sizeof(config->bitnet_gguf) - 1);
    }
    if ((env_val = getenv("AURORA_BITNET_DIR"))) {
        strncpy(config->bitnet_dir, env_val, sizeof(config->bitnet_dir) - 1);
    }
    if ((env_val = getenv("AURORA_LLAMA_CLI"))) {
        strncpy(config->llama_cli, env_val, sizeof(config->llama_cli) - 1);
    }
    if ((env_val = getenv("AURORA_TEXT_MODEL"))) {
        strncpy(config->text_model, env_val, sizeof(config->text_model) - 1);
    }
    if ((env_val = getenv("AURORA_DEEPSEEK_VL_MODEL"))) {
        strncpy(config->deepseek_vl_model, env_val, sizeof(config->deepseek_vl_model) - 1);
    }
    if ((env_val = getenv("AURORA_WIKI_DATASET_DIR"))) {
        strncpy(config->wiki_dataset_dir, env_val, sizeof(config->wiki_dataset_dir) - 1);
    }
    
    // Navigation settings
    if ((env_val = getenv("AURORA_NAVIGATION_ENABLED"))) {
        config->navigation_enabled = (strcmp(env_val, "1") == 0 || strcasecmp(env_val, "true") == 0);
    }
    if ((env_val = getenv("AURORA_NAV_TAU"))) {
        config->nav_tau = (float)atof(env_val);
    }
    if ((env_val = getenv("AURORA_NAV_MAX_DEPTH"))) {
        config->nav_max_depth = (int32_t)atoi(env_val);
    }
    if ((env_val = getenv("AURORA_NAV_MAX_NODES"))) {
        config->nav_max_nodes = (int32_t)atoi(env_val);
    }
    if ((env_val = getenv("AURORA_NAV_WINDOW_SIZE"))) {
        config->nav_window_size = (int32_t)atoi(env_val);
    }
    if ((env_val = getenv("AURORA_NAV_K_READ"))) {
        config->nav_k_read = (int32_t)atoi(env_val);
    }
    
    // GPU metrics
    if ((env_val = getenv("AURORA_GPU_METRICS_ENABLED"))) {
        config->gpu_metrics_enabled = (strcmp(env_val, "1") == 0 || strcasecmp(env_val, "true") == 0);
    }
    
    // Build settings
    if ((env_val = getenv("AURORA_NVCC_CCBIN"))) {
        strncpy(config->nvcc_ccbin, env_val, sizeof(config->nvcc_ccbin) - 1);
    }
    if ((env_val = getenv("AURORA_NVPTX_TOOLCHAIN"))) {
        strncpy(config->nvptx_toolchain, env_val, sizeof(config->nvptx_toolchain) - 1);
    }
}

int aurora_config_load_from_json(struct aurora_config* config, const char* json_path) {
    if (!config || !json_path) return -1;
    
    // Initialize defaults first
    aurora_config_init_defaults(config);
    
    FILE* f = fopen(json_path, "r");
    if (!f) {
        // If file doesn't exist, use defaults + environment
        aurora_config_load_from_env(config);
        return 0;  // Not an error - defaults are acceptable
    }
    
    // Read entire file (simple approach for small config files)
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size <= 0 || size > 10240) {  // Max 10KB
        fclose(f);
        aurora_config_load_from_env(config);
        return 0;
    }
    
    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fclose(f);
        aurora_config_load_from_env(config);
        return 0;
    }
    
    size_t read = fread(buffer, 1, size, f);
    buffer[read] = '\0';
    fclose(f);
    
    // Simple JSON parsing (minimal implementation)
    const char* p = buffer;
    skip_whitespace(&p);
    if (*p != '{') {
        free(buffer);
        aurora_config_load_from_env(config);
        return 0;
    }
    p++;
    
    // Parse paths section
    if (find_key(&p, "paths")) {
        skip_whitespace(&p);
        if (*p == '{') {
            p++;
            while (*p && *p != '}') {
                skip_whitespace(&p);
                if (*p == '"') {
                    if (find_key(&p, "bitnet_gguf")) {
                        parse_string(&p, config->bitnet_gguf, sizeof(config->bitnet_gguf));
                    } else if (find_key(&p, "bitnet_dir")) {
                        parse_string(&p, config->bitnet_dir, sizeof(config->bitnet_dir));
                    } else if (find_key(&p, "llama_cli")) {
                        parse_string(&p, config->llama_cli, sizeof(config->llama_cli));
                    } else if (find_key(&p, "text_model")) {
                        parse_string(&p, config->text_model, sizeof(config->text_model));
                    } else if (find_key(&p, "deepseek_vl_model")) {
                        parse_string(&p, config->deepseek_vl_model, sizeof(config->deepseek_vl_model));
                    } else if (find_key(&p, "wiki_dataset_dir")) {
                        parse_string(&p, config->wiki_dataset_dir, sizeof(config->wiki_dataset_dir));
                    } else {
                        // Skip unknown key
                        while (*p && *p != ',' && *p != '}') p++;
                    }
                }
                skip_whitespace(&p);
                if (*p == ',') p++;
            }
        }
    }
    
    // Parse navigation section
    p = buffer;  // Reset for simplicity (could optimize)
    if (find_key(&p, "navigation")) {
        skip_whitespace(&p);
        if (*p == '{') {
            p++;
            while (*p && *p != '}') {
                skip_whitespace(&p);
                if (find_key(&p, "enabled")) {
                    parse_bool(&p, &config->navigation_enabled);
                } else if (find_key(&p, "tau")) {
                    parse_number(&p, &config->nav_tau, NULL);
                } else if (find_key(&p, "max_depth")) {
                    parse_number(&p, NULL, &config->nav_max_depth);
                } else if (find_key(&p, "max_nodes")) {
                    parse_number(&p, NULL, &config->nav_max_nodes);
                } else if (find_key(&p, "window_size")) {
                    parse_number(&p, NULL, &config->nav_window_size);
                } else if (find_key(&p, "k_read")) {
                    parse_number(&p, NULL, &config->nav_k_read);
                } else {
                    while (*p && *p != ',' && *p != '}') p++;
                }
                skip_whitespace(&p);
                if (*p == ',') p++;
            }
        }
    }
    
    // Parse gpu_metrics section
    p = buffer;
    if (find_key(&p, "gpu_metrics")) {
        skip_whitespace(&p);
        if (*p == '{') {
            p++;
            if (find_key(&p, "enabled")) {
                parse_bool(&p, &config->gpu_metrics_enabled);
            }
        }
    }
    
    free(buffer);
    
    // Override with environment variables (env takes precedence)
    aurora_config_load_from_env(config);
    
    return 0;
}

int aurora_config_save_to_json(const struct aurora_config* config, const char* json_path) {
    if (!config || !json_path) return -1;
    
    FILE* f = fopen(json_path, "w");
    if (!f) return -1;
    
    fprintf(f, "{\n");
    fprintf(f, "  \"paths\": {\n");
    fprintf(f, "    \"bitnet_gguf\": \"%s\",\n", config->bitnet_gguf);
    fprintf(f, "    \"bitnet_dir\": \"%s\",\n", config->bitnet_dir);
    fprintf(f, "    \"llama_cli\": \"%s\",\n", config->llama_cli);
    fprintf(f, "    \"text_model\": \"%s\",\n", config->text_model);
    fprintf(f, "    \"deepseek_vl_model\": \"%s\",\n", config->deepseek_vl_model);
    fprintf(f, "    \"wiki_dataset_dir\": \"%s\"\n", config->wiki_dataset_dir);
    fprintf(f, "  },\n");
    fprintf(f, "  \"navigation\": {\n");
    fprintf(f, "    \"enabled\": %s,\n", config->navigation_enabled ? "true" : "false");
    fprintf(f, "    \"tau\": %.2f,\n", config->nav_tau);
    fprintf(f, "    \"max_depth\": %d,\n", config->nav_max_depth);
    fprintf(f, "    \"max_nodes\": %d,\n", config->nav_max_nodes);
    fprintf(f, "    \"window_size\": %d,\n", config->nav_window_size);
    fprintf(f, "    \"k_read\": %d\n", config->nav_k_read);
    fprintf(f, "  },\n");
    fprintf(f, "  \"gpu_metrics\": {\n");
    fprintf(f, "    \"enabled\": %s\n", config->gpu_metrics_enabled ? "true" : "false");
    fprintf(f, "  },\n");
    fprintf(f, "  \"build\": {\n");
    fprintf(f, "    \"nvcc_ccbin\": \"%s\",\n", config->nvcc_ccbin);
    fprintf(f, "    \"nvptx_toolchain\": \"%s\"\n", config->nvptx_toolchain);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    
    fclose(f);
    return 0;
}

const char* aurora_config_get(const struct aurora_config* config, const char* key) {
    if (!config || !key) return NULL;
    
    // Paths
    if (strcmp(key, "bitnet_gguf") == 0) return config->bitnet_gguf;
    if (strcmp(key, "bitnet_dir") == 0) return config->bitnet_dir;
    if (strcmp(key, "llama_cli") == 0) return config->llama_cli;
    if (strcmp(key, "text_model") == 0) return config->text_model;
    if (strcmp(key, "deepseek_vl_model") == 0) return config->deepseek_vl_model;
    if (strcmp(key, "wiki_dataset_dir") == 0) return config->wiki_dataset_dir;
    
    // Navigation (return NULL for non-string values - use specific getters)
    // For now, return NULL for numeric/bool values
    return NULL;
}

int aurora_config_set(struct aurora_config* config, const char* key, const char* value) {
    if (!config || !key || !value) return -1;
    
    // Paths
    if (strcmp(key, "bitnet_gguf") == 0) {
        strncpy(config->bitnet_gguf, value, sizeof(config->bitnet_gguf) - 1);
        return 0;
    }
    if (strcmp(key, "bitnet_dir") == 0) {
        strncpy(config->bitnet_dir, value, sizeof(config->bitnet_dir) - 1);
        return 0;
    }
    if (strcmp(key, "llama_cli") == 0) {
        strncpy(config->llama_cli, value, sizeof(config->llama_cli) - 1);
        return 0;
    }
    if (strcmp(key, "text_model") == 0) {
        strncpy(config->text_model, value, sizeof(config->text_model) - 1);
        return 0;
    }
    if (strcmp(key, "deepseek_vl_model") == 0) {
        strncpy(config->deepseek_vl_model, value, sizeof(config->deepseek_vl_model) - 1);
        return 0;
    }
    if (strcmp(key, "wiki_dataset_dir") == 0) {
        strncpy(config->wiki_dataset_dir, value, sizeof(config->wiki_dataset_dir) - 1);
        return 0;
    }
    
    // Navigation (for numeric/bool, would need separate setters)
    // For now, return -1 for unsupported keys
    return -1;
}
