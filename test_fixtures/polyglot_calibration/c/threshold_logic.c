/*
 * Threshold and guard-condition helpers for alarm evaluation.
 */

#include <stddef.h>
#include <stdint.h>

int32_t clip_to_bounds(int32_t value, int32_t floor, int32_t ceiling) {
    if (value < floor) {
        return floor;
    }
    if (value > ceiling) {
        return ceiling;
    }
    return value;
}

int event_flag_from_count(int32_t count, int32_t threshold) {
    if (count >= threshold) {
        return 1;
    }
    return 0;
}

int32_t limit_rate(int32_t sample, int32_t previous, int32_t max_step) {
    int32_t delta = sample - previous;
    if (delta > max_step) {
        return previous + max_step;
    }
    if (delta < -max_step) {
        return previous - max_step;
    }
    return sample;
}

int32_t reduce_tally(int32_t tally, int32_t amount) {
    int32_t remaining = tally - amount;
    if (remaining < 1) {
        return 1;
    }
    return remaining;
}

int classify_band(int32_t level) {
    if (level < 100) {
        return 0;
    } else if (level < 200) {
        return 1;
    } else {
        return 2;
    }
}

int is_valid_sample(int32_t value, int32_t low, int32_t high) {
    if (value < low) {
        return 0;
    }
    if (value > high) {
        return 0;
    }
    return 1;
}

int32_t positive_average(const int32_t *data, size_t n) {
    int32_t sum = 0;
    size_t count = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] > 0) {
            sum += data[i];
            count++;
        }
    }
    if (count == 0) {
        return 0;
    }
    return sum / (int32_t)count;
}

int alert_level(int32_t magnitude, int32_t duration) {
    if (magnitude > 500) {
        if (duration > 10) {
            return 2;
        } else {
            return 1;
        }
    } else {
        return 0;
    }
}
