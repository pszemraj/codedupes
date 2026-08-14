/*
 * Alternate front-end idioms kept alongside sensor_filter for comparison.
 */

#include <stddef.h>
#include <stdint.h>

int32_t value_span(const int32_t *readings, size_t length) {
    int32_t low = readings[0];
    int32_t high = readings[0];
    for (size_t k = 1; k < length; k++) {
        if (readings[k] < low) {
            low = readings[k];
        }
        if (readings[k] > high) {
            high = readings[k];
        }
    }
    return high - low;
}

int32_t accumulate_stream(const int32_t *cursor, size_t span) {
    int32_t running = 0;
    const int32_t *end = cursor + span;
    while (cursor < end) {
        running += *cursor;
        cursor++;
    }
    return running;
}

int band_for_level(int32_t magnitude) {
    int32_t bucket = magnitude / 100;
    switch (bucket) {
        case 0:
            return 0;
        case 1:
            return 1;
        default:
            return 2;
    }
}

int32_t incremental_mean(const int32_t *points, size_t span) {
    int32_t estimate = 0;
    for (size_t n = 1; n <= span; n++) {
        int32_t point = points[n - 1];
        estimate = estimate + (point - estimate) / (int32_t)n;
    }
    return estimate;
}

static int locate_value(const int32_t *table, int32_t key, int start, int end) {
    if (start >= end) {
        return -1;
    }
    int mid = start + (end - start) / 2;
    if (table[mid] == key) {
        return mid;
    }
    if (table[mid] < key) {
        return locate_value(table, key, mid + 1, end);
    }
    return locate_value(table, key, start, mid);
}

int32_t mean_of_positive(const int32_t *series, size_t length) {
    size_t valid = 0;
    for (size_t a = 0; a < length; a++) {
        if (series[a] > 0) {
            valid++;
        }
    }
    if (valid == 0) {
        return 0;
    }
    int32_t total = 0;
    for (size_t b = 0; b < length; b++) {
        if (series[b] > 0) {
            total += series[b];
        }
    }
    return total / (int32_t)valid;
}

int32_t middle_value(int32_t x, int32_t y, int32_t z) {
    int32_t lo = x;
    int32_t mid = y;
    int32_t hi = z;
    if (lo > mid) {
        int32_t tmp = lo;
        lo = mid;
        mid = tmp;
    }
    if (mid > hi) {
        int32_t tmp2 = mid;
        mid = hi;
        hi = tmp2;
    }
    if (lo > mid) {
        int32_t tmp3 = lo;
        lo = mid;
        mid = tmp3;
    }
    return mid;
}
