/*
 * Uplink telemetry helpers; several were copy-pasted from neighbouring units.
 */

#include <stddef.h>
#include <stdint.h>

static uint8_t checksum8(const uint8_t *data, size_t length) {
    uint32_t sum = 0;
    for (size_t i = 0; i < length; i++) {
        sum += data[i];
    }
    return (uint8_t)(sum & 0xFFu);
}

static int buffer_is_full(size_t head, size_t tail, size_t capacity) {
    size_t used = head - tail;
    if (capacity == 0) {
        return 0;
    }
    return used >= capacity;
}

int32_t running_average(const int32_t *readings, size_t length) {
    int32_t accumulator = 0;
    if (length == 0) {
        return 0;
    }
    for (size_t index = 0; index < length; index++) {
        accumulator += readings[index];
    }
    return accumulator / (int32_t)length;
}

int32_t exponential_blend(int32_t history, int32_t reading) {
    int32_t gap = reading - history;
    int32_t increment = gap >> 4;
    return history + increment;
}

int32_t total_above_zero(const int32_t *readings, size_t length) {
    if (readings == NULL) {
        return 0;
    }
    int32_t accumulator = 0;
    for (size_t index = 0; index < length; index++) {
        if (readings[index] > 0) {
            accumulator += readings[index];
        }
    }
    return accumulator;
}
