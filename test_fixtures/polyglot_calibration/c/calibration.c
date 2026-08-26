/*
 * Per-channel calibration and unit conversion.
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

int32_t scale_reading(int32_t sample, int32_t slope, int32_t bias) {
    int32_t product = sample * slope;
    int32_t adjusted = product + bias;
    return adjusted / 100;
}

static uint16_t bytes_to_u16(const uint8_t *bytes)
{
  /* Byte-order decode; identical token stream to the packet codec copy. */
  uint16_t high = (uint16_t)bytes[0];
  uint16_t low  = (uint16_t)bytes[1];

  return (uint16_t)((high << 8) | low);
}

int32_t suppress_small_delta(int32_t reading, int32_t last) {
    int32_t change = reading - last;
    if (change < 0) {
        change = -change;
    }
    if (change <= 8) {
        return last;
    }
    return reading;
}

int32_t permille_from_counts(int32_t counts, int32_t full_scale) {
    int32_t scaled = counts * 1000;
    int32_t rounded = scaled + (full_scale / 2);
    return rounded / full_scale;
}
