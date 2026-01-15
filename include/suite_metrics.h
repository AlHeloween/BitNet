#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Suite-level teacher metrics and promotion gating
// Stage 3: Teacher/verification loop formalization

// Confusion matrix for binary classification (promote vs don't promote)
typedef struct {
    int32_t tp;  // True positives
    int32_t fp;  // False positives
    int32_t tn;  // True negatives
    int32_t fn;  // False negatives
} confusion_matrix_t;

// Compute confusion matrix metrics
float confusion_matrix_precision(const confusion_matrix_t* cm);
float confusion_matrix_recall(const confusion_matrix_t* cm);
float confusion_matrix_f1_score(const confusion_matrix_t* cm);
float confusion_matrix_accuracy(const confusion_matrix_t* cm);

// Suite metrics structure
typedef struct {
    confusion_matrix_t confusion_matrix;
    int32_t* promoted_indices;      // Array of promoted index IDs
    int32_t n_promoted;
    int32_t* blocked_indices;       // Array of blocked index IDs
    int32_t n_blocked;
    int32_t train_events;
    int32_t test_events;
    int32_t eligible_indices;
    float promote_threshold;
    int32_t promote_min_count;
} suite_metrics_t;

// Training/test event structure
typedef struct {
    int32_t idx;      // Memory index
    bool match;       // Whether teacher matched (correct)
} suite_event_t;

// Compute suite metrics from train/test events
// Returns 0 on success, -1 on failure
// Caller must free suite_metrics using suite_metrics_free()
int compute_suite_metrics(
    const suite_event_t* train_events,
    int32_t n_train_events,
    const suite_event_t* test_events,
    int32_t n_test_events,
    float promote_threshold,
    int32_t promote_min_count,
    suite_metrics_t* out
);

// Free suite metrics
void suite_metrics_free(suite_metrics_t* metrics);

// Compute token match rate
float compute_token_match_rate(
    const int32_t* baseline_tokens,
    int32_t n_baseline,
    const int32_t* generated_tokens,
    int32_t n_generated
);

#ifdef __cplusplus
}
#endif
