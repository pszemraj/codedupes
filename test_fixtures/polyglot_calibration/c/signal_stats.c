/*
 * Descriptive statistics over fixed-length windows of samples.
 */

#include <stddef.h>
#include <stdint.h>

int32_t compute_mean(const int32_t *samples, size_t count) {
    int32_t total = 0;
    if (count == 0) {
        return 0;
    }
    for (size_t i = 0; i < count; i++) {
        total += samples[i];
    }
    return total / (int32_t)count;
}

size_t find_peak_index(const int32_t *samples, size_t count) {
    size_t best = 0;
    for (size_t i = 1; i < count; i++) {
        if (samples[i] > samples[best]) {
            best = i;
        }
    }
    return best;
}

/* Reformatted copy of the sensor_filter helper; kept local to this unit. */
static int32_t moving_sum(const int32_t *window, size_t length)
{
  int32_t total = 0;

  /* Accumulate the whole window; callers guarantee no overflow. */
  for (size_t i = 0; i < length; i++)
  {
    total += window[i];
  }

  return total;
}

int32_t sum_positive(const int32_t *samples, size_t count) {
    int32_t total = 0;
    for (size_t i = 0; i < count; i++) {
        if (samples[i] > 0) {
            total += samples[i];
        }
    }
    return total;
}

size_t find_trough_index(const int32_t *samples, size_t count) {
    size_t best = 0;
    int32_t lowest = samples[0];
    for (size_t i = 1; i < count; i++) {
        if (samples[i] < lowest) {
            lowest = samples[i];
            best = i;
        }
    }
    return best;
}
