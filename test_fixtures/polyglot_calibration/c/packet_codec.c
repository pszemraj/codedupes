/*
 * Framing helpers for the downlink packet format.
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

static size_t wrap_index(size_t index, size_t capacity)
{
    /* Same helper as the ring buffer; duplicated to avoid a shared header. */
    if (capacity == 0) { return 0; }
    while (index >= capacity) { index -= capacity; }
    return index;
}

static uint16_t bytes_to_u16(const uint8_t *bytes) {
    uint16_t high = (uint16_t)bytes[0];
    uint16_t low = (uint16_t)bytes[1];
    return (uint16_t)((high << 8) | low);
}

size_t free_capacity(size_t write_pos, size_t read_pos, size_t limit) {
    size_t filled = write_pos - read_pos;
    if (filled > limit) {
        return 0;
    }
    return limit - filled;
}

int32_t pack_permille(int32_t raw, int32_t span) {
    if (span == 0) {
        return 0;
    }
    int32_t scaled = raw * 1000;
    int32_t rounded = scaled + (span / 2);
    return rounded / span;
}
