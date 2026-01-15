// Test suite for Aurora Navigation paradigm (Stage 4)
// Tests C++ implementation of Navigation paradigm functions

#include "llama-aurora-navigation.h"
#include "aurora_memory_bank.h"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>

// Test helper: Create a simple dual quaternion
static void create_test_dual_quat(float* dq, float w, float x, float y, float z) {
    // Primal part (rotation quaternion)
    dq[0] = w;  // w
    dq[1] = x;  // x
    dq[2] = y;  // y
    dq[3] = z;  // z
    // Dual part (translation quaternion) - set to zero for simplicity
    dq[4] = 0.0f;
    dq[5] = 0.0f;
    dq[6] = 0.0f;
    dq[7] = 0.0f;
}

// Test 1: Navigation params initialization
static bool test_navigation_params_init() {
    printf("[TEST] Navigation params initialization...\n");
    
    struct llama_aurora_navigation_params params;
    llama_aurora_navigation_params_init(&params);
    
    assert(params.enable_navigation_paradigm == false);
    assert(fabs(params.tau - 0.1f) < 1e-6f);
    assert(params.max_depth == 7);
    assert(params.max_nodes_per_level == 4);
    assert(params.apply_temporal_decay == false);
    assert(fabs(params.current_time - 0.0f) < 1e-6f);
    
    printf("  ✓ Navigation params initialized correctly\n");
    return true;
}

// Test 2: Extract primal similarity
static bool test_extract_primal_similarity() {
    printf("[TEST] Extract primal similarity...\n");
    
    // Create a test score dual quaternion
    // For normalized quaternion with w=1.0, similarity should be 1.0
    float score_dq_data[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    struct ggml_tensor score_dq;
    memset(&score_dq, 0, sizeof(score_dq));
    score_dq.type = GGML_TYPE_F32;
    score_dq.ne[0] = 8;
    score_dq.data = score_dq_data;
    
    float similarity = llama_aurora_extract_primal_similarity(&score_dq);
    
    // For w=1.0, normalized similarity = (1.0 + 1.0) / 2.0 = 1.0
    assert(fabs(similarity - 1.0f) < 1e-6f);
    
    printf("  ✓ Primal similarity extracted correctly: %.3f\n", similarity);
    return true;
}

// Test 3: Extract dual velocity
static bool test_extract_dual_velocity() {
    printf("[TEST] Extract dual velocity...\n");
    
    // Create a test score dual quaternion with dual part
    float score_dq_data[8] = {
        1.0f, 0.0f, 0.0f, 0.0f,  // Primal: w=1.0
        3.0f, 4.0f, 0.0f, 0.0f   // Dual: (3, 4, 0, 0) -> magnitude = 5.0
    };
    
    struct ggml_tensor score_dq;
    memset(&score_dq, 0, sizeof(score_dq));
    score_dq.type = GGML_TYPE_F32;
    score_dq.ne[0] = 8;
    score_dq.data = score_dq_data;
    
    float velocity = llama_aurora_extract_dual_velocity(&score_dq);
    
    // ||zd|| = sqrt(3² + 4² + 0² + 0²) = 5.0
    assert(fabs(velocity - 5.0f) < 1e-6f);
    
    printf("  ✓ Dual velocity extracted correctly: %.3f\n", velocity);
    return true;
}

// Test 4: Fractal drill-down (basic)
static bool test_fractal_drill_down_basic() {
    printf("[TEST] Fractal drill-down (basic)...\n");
    
    // Create memory banks
    aurora_memory_banks_t* banks = aurora_memory_banks_create(
        16,  // n_clusters
        8,   // dim (dual quaternion)
        1,   // depth
        1234, // seed
        false // allow_quarantine_read
    );
    
    if (!banks) {
        printf("  ✗ Failed to create memory banks\n");
        return false;
    }
    
    // Add a test entry
    bool added = aurora_memory_bank_add(
        banks->verified,
        "Test entry for navigation",
        AURORA_INFO_EXACT,
        NULL,  // Use same text for embedding
        NULL   // Compute MD5 automatically
    );
    
    if (!added) {
        printf("  ✗ Failed to add test entry\n");
        aurora_memory_banks_free(banks);
        return false;
    }
    
    // Create GGML context
    struct ggml_context* ctx = ggml_init({});
    if (!ctx) {
        printf("  ✗ Failed to create GGML context\n");
        aurora_memory_banks_free(banks);
        return false;
    }
    
    // Create query dual quaternion
    float query_data[8];
    create_test_dual_quat(query_data, 1.0f, 0.0f, 0.0f, 0.0f);
    
    struct ggml_tensor* query_dq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    memcpy(query_dq->data, query_data, 8 * sizeof(float));
    
    // Set up navigation params
    struct llama_aurora_navigation_params nav_params;
    llama_aurora_navigation_params_init(&nav_params);
    nav_params.enable_navigation_paradigm = true;
    nav_params.tau = 0.1f;
    nav_params.max_depth = 7;
    nav_params.max_nodes_per_level = 4;
    
    // Call fractal drill-down
    struct ggml_tensor* result = llama_aurora_fractal_drill_down(
        ctx,
        query_dq,
        banks,
        &nav_params
    );
    
    bool success = (result != NULL);
    
    if (success) {
        printf("  ✓ Fractal drill-down returned result\n");
    } else {
        printf("  ✗ Fractal drill-down returned NULL\n");
    }
    
    // Cleanup
    ggml_free(ctx);
    aurora_memory_banks_free(banks);
    
    return success;
}

int main() {
    printf("========================================\n");
    printf("AURORA NAVIGATION PARADIGM C++ TESTS\n");
    printf("========================================\n\n");
    
    int passed = 0;
    int total = 0;
    
    // Run tests
    total++; if (test_navigation_params_init()) passed++;
    total++; if (test_extract_primal_similarity()) passed++;
    total++; if (test_extract_dual_velocity()) passed++;
    total++; if (test_fractal_drill_down_basic()) passed++;
    
    printf("\n========================================\n");
    printf("RESULTS: %d/%d tests passed\n", passed, total);
    printf("========================================\n");
    
    return (passed == total) ? 0 : 1;
}
