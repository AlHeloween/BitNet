#include "suite_metrics.h"

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <vector>
#include <set>
#include <cmath>

// Compute confusion matrix metrics
float confusion_matrix_precision(const confusion_matrix_t* cm) {
    int32_t denom = cm->tp + cm->fp;
    return (denom > 0) ? (float(cm->tp) / float(denom)) : 0.0f;
}

float confusion_matrix_recall(const confusion_matrix_t* cm) {
    int32_t denom = cm->tp + cm->fn;
    return (denom > 0) ? (float(cm->tp) / float(denom)) : 0.0f;
}

float confusion_matrix_f1_score(const confusion_matrix_t* cm) {
    float p = confusion_matrix_precision(cm);
    float r = confusion_matrix_recall(cm);
    float denom = p + r;
    return (denom > 0.0f) ? (2.0f * p * r / denom) : 0.0f;
}

float confusion_matrix_accuracy(const confusion_matrix_t* cm) {
    int32_t total = cm->tp + cm->fp + cm->tn + cm->fn;
    return (total > 0) ? (float(cm->tp + cm->tn) / float(total)) : 0.0f;
}

// Compute suite metrics
int compute_suite_metrics(
    const suite_event_t* train_events,
    int32_t n_train_events,
    const suite_event_t* test_events,
    int32_t n_test_events,
    float promote_threshold,
    int32_t promote_min_count,
    suite_metrics_t* out
) {
    if (!train_events || !test_events || !out || n_train_events < 0 || n_test_events < 0) {
        return -1;
    }
    
    // Aggregate training events by index
    std::map<int32_t, std::pair<int32_t, int32_t>> per_idx;  // index -> (total, matches)
    
    for (int32_t i = 0; i < n_train_events; i++) {
        int32_t idx = train_events[i].idx;
        bool match = train_events[i].match;
        
        auto& counts = per_idx[idx];
        counts.first += 1;  // total
        if (match) {
            counts.second += 1;  // matches
        }
    }
    
    // Determine safe set (indices to promote)
    std::vector<int32_t> promoted;
    std::vector<int32_t> blocked;
    
    for (const auto& entry : per_idx) {
        int32_t idx = entry.first;
        int32_t total = entry.second.first;
        int32_t matches = entry.second.second;
        float rate = float(matches) / float(std::max(1, total));
        
        if (total >= promote_min_count && rate >= promote_threshold) {
            promoted.push_back(idx);
        } else {
            blocked.push_back(idx);
        }
    }
    
    std::sort(promoted.begin(), promoted.end());
    std::sort(blocked.begin(), blocked.end());
    
    // Compute confusion matrix on test set
    confusion_matrix_t cm = {0, 0, 0, 0};
    std::set<int32_t> safe_set(promoted.begin(), promoted.end());
    
    for (int32_t i = 0; i < n_test_events; i++) {
        int32_t idx = test_events[i].idx;
        bool truth = test_events[i].match;
        bool pred = safe_set.count(idx) > 0;
        
        if (pred && truth) {
            cm.tp++;
        } else if (pred && !truth) {
            cm.fp++;
        } else if (!pred && !truth) {
            cm.tn++;
        } else {  // !pred && truth
            cm.fn++;
        }
    }
    
    // Allocate output structure
    memset(out, 0, sizeof(suite_metrics_t));
    out->confusion_matrix = cm;
    out->train_events = n_train_events;
    out->test_events = n_test_events;
    out->eligible_indices = static_cast<int32_t>(per_idx.size());
    out->promote_threshold = promote_threshold;
    out->promote_min_count = promote_min_count;
    
    // Copy promoted indices
    out->n_promoted = static_cast<int32_t>(promoted.size());
    if (out->n_promoted > 0) {
        out->promoted_indices = static_cast<int32_t*>(malloc(out->n_promoted * sizeof(int32_t)));
        if (!out->promoted_indices) {
            return -1;
        }
        std::copy(promoted.begin(), promoted.end(), out->promoted_indices);
    }
    
    // Copy blocked indices
    out->n_blocked = static_cast<int32_t>(blocked.size());
    if (out->n_blocked > 0) {
        out->blocked_indices = static_cast<int32_t*>(malloc(out->n_blocked * sizeof(int32_t)));
        if (!out->blocked_indices) {
            suite_metrics_free(out);
            return -1;
        }
        std::copy(blocked.begin(), blocked.end(), out->blocked_indices);
    }
    
    return 0;
}

// Free suite metrics
void suite_metrics_free(suite_metrics_t* metrics) {
    if (!metrics) return;
    
    if (metrics->promoted_indices) {
        free(metrics->promoted_indices);
        metrics->promoted_indices = nullptr;
    }
    
    if (metrics->blocked_indices) {
        free(metrics->blocked_indices);
        metrics->blocked_indices = nullptr;
    }
    
    metrics->n_promoted = 0;
    metrics->n_blocked = 0;
}

// Compute token match rate
float compute_token_match_rate(
    const int32_t* baseline_tokens,
    int32_t n_baseline,
    const int32_t* generated_tokens,
    int32_t n_generated
) {
    if (!baseline_tokens || !generated_tokens || n_baseline <= 0 || n_generated <= 0) {
        return 0.0f;
    }
    
    int32_t n = (n_baseline < n_generated) ? n_baseline : n_generated;
    if (n == 0) {
        return 0.0f;
    }
    
    int32_t matches = 0;
    for (int32_t i = 0; i < n; i++) {
        if (baseline_tokens[i] == generated_tokens[i]) {
            matches++;
        }
    }
    
    return float(matches) / float(n);
}

#ifdef __cplusplus
}
#endif
