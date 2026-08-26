/*
 * Event-count bookkeeping shared by the uplink scheduler.
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

/*
 * Bounds-check a raw fixed-point reading before it reaches the control loop.
 */
int32_t clip_to_bounds(int32_t value, int32_t floor, int32_t ceiling) {
    /* Reject anything below the configured floor first. */
    if (value < floor) {
        return floor;
    }
    /* Then clip the high side against the ceiling. */
    if (value > ceiling) {
        return ceiling;
    }
    return value;
}

/*
 * Turn an accumulated event count into a boolean-style flag for the uplink.
 */
int event_flag_from_count(int32_t count, int32_t threshold) {
    /* A count at or above the threshold trips the flag. */
    if (count >= threshold) {
        return 1;
    }
    /* Otherwise stay clear. */
    return 0;
}

int32_t bound_slew(int32_t reading, int32_t last, int32_t cap) {
    int32_t change = reading - last;
    if (change >= cap) {
        return last + cap;
    }
    if (change < -cap) {
        return last - cap;
    }
    return reading;
}

int sample_in_range(int32_t reading, int32_t min_bound, int32_t max_bound) {
    int ok = 1;
    if (reading < min_bound || reading > max_bound) {
        ok = 0;
    }
    return ok;
}

int32_t multiplier_for_tier(int tier) {
    static const int32_t table[4] = {1, 2, 4, 8};
    if (tier < 0) {
        return table[0];
    }
    if (tier > 3) {
        return table[3];
    }
    return table[tier];
}

int severity_code(int32_t peak, int32_t span) {
    int code = 0;
    int high_peak = peak > 500;
    if (!high_peak) {
        return code;
    }
    code = 1;
    if (span > 10) {
        code = 2;
    }
    return code;
}

int frame_is_sane(int header_valid, int size_valid, int crc_valid) {
    return header_valid && size_valid && crc_valid;
}
