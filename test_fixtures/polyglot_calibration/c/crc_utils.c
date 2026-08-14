/*
 * Checksum and level-mapping helpers kept separate from the packet framer.
 */

#include <stddef.h>
#include <stdint.h>

static uint8_t checksum_xor8(const uint8_t *data, size_t length) {
    uint8_t result = 0;
    for (size_t i = 0; i < length; i++) {
        result ^= data[i];
    }
    return result;
}

int32_t weighted_delta(int32_t current, int32_t baseline, int32_t weight) {
    int32_t diff = current - baseline;
    int32_t scaled = diff * weight;
    return scaled / 16;
}

int32_t decay_counter(int32_t counter, int32_t step) {
    int32_t next = counter - step;
    if (next < 0) {
        return 0;
    }
    return next;
}

int32_t gain_for_level(int level) {
    switch (level) {
        case 0:
            return 1;
        case 1:
            return 2;
        case 2:
            return 4;
        default:
            return 8;
    }
}

int any_out_of_bounds(const int32_t *samples, size_t count, int32_t limit) {
    for (size_t i = 0; i < count; i++) {
        if (samples[i] > limit || samples[i] < -limit) {
            return 1;
        }
    }
    return 0;
}

int packet_is_valid(int has_header, int has_length, int has_checksum) {
    if (!has_header) {
        return 0;
    }
    if (!has_length) {
        return 0;
    }
    if (!has_checksum) {
        return 0;
    }
    return 1;
}

int popcount_flags(uint32_t flags) {
    int tally = 0;
    while (flags) {
        flags &= (flags - 1);
        tally++;
    }
    return tally;
}
