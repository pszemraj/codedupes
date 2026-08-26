/*
 * Fixed-point front-end filters for raw ADC sample streams.
 */

#include <stddef.h>
#include <stdint.h>

static int32_t clamp_i32(int32_t value, int32_t low, int32_t high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static int32_t moving_sum(const int32_t *window, size_t length) {
    int32_t total = 0;
    for (size_t i = 0; i < length; i++) {
        total += window[i];
    }
    return total;
}

int32_t convert_sample(int32_t raw, int32_t gain, int32_t offset) {
    int32_t scaled = raw * gain;
    int32_t shifted = scaled + offset;
    return shifted / 100;
}

int32_t apply_deadband(int32_t sample, int32_t previous) {
    int32_t delta = sample - previous;
    if (delta < 0) {
        delta = -delta;
    }
    if (delta < 8) {
        return previous;
    }
    return sample;
}

int32_t smooth_ema(int32_t previous, int32_t sample) {
    int32_t difference = sample - previous;
    int32_t step = difference >> 3;
    return previous + step;
}
