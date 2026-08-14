/*
 * Windowed array statistics used by the telemetry summarizer.
 */

#include <stddef.h>
#include <stdint.h>

int32_t sample_range(const int32_t *samples, size_t count) {
    int32_t lo = samples[0];
    int32_t hi = samples[0];
    for (size_t i = 1; i < count; i++) {
        if (samples[i] < lo) {
            lo = samples[i];
        }
        if (samples[i] > hi) {
            hi = samples[i];
        }
    }
    return hi - lo;
}

/*
 * Scale a reading's deviation from baseline by a tunable weight.
 */
int32_t weighted_delta(int32_t current, int32_t baseline, int32_t weight) {
    /* Distance from the calibrated baseline. */
    int32_t diff = current - baseline;
    /* Apply the caller-selected weighting factor. */
    int32_t scaled = diff * weight;
    /* Fixed divisor keeps this integer-only. */
    return scaled / 16;
}

int32_t total_samples(const int32_t *data, size_t count) {
    int32_t sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += data[i];
    }
    return sum;
}

int32_t batch_mean(const int32_t *values, size_t length) {
    int32_t total = 0;
    for (size_t i = 0; i < length; i++) {
        total += values[i];
    }
    if (length == 0) {
        return 0;
    }
    return total / (int32_t)length;
}

int find_index(const int32_t *sorted, size_t length, int32_t target) {
    size_t low = 0;
    size_t high = length;
    while (low < high) {
        size_t pivot = low + (high - low) / 2;
        if (sorted[pivot] == target) {
            return (int)pivot;
        }
        if (sorted[pivot] < target) {
            low = pivot + 1;
        } else {
            high = pivot;
        }
    }
    return -1;
}

int count_set_bits(uint32_t mask) {
    int total = 0;
    for (int i = 0; i < 32; i++) {
        if (mask & (1u << i)) {
            total++;
        }
    }
    return total;
}

int detect_overrange(const int32_t *readings, size_t length, int32_t bound) {
    int flagged = 0;
    for (size_t k = 0; k < length; k++) {
        int32_t magnitude = readings[k] < 0 ? -readings[k] : readings[k];
        if (magnitude > bound) {
            flagged = flagged | 1;
        }
    }
    return flagged;
}

int32_t median_of_three(int32_t a, int32_t b, int32_t c) {
    if (a > b) {
        if (b > c) {
            return b;
        } else if (a > c) {
            return c;
        } else {
            return a;
        }
    } else {
        if (a > c) {
            return a;
        } else if (b > c) {
            return c;
        } else {
            return b;
        }
    }
}
