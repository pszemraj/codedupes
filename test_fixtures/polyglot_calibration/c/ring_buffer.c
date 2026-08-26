/*
 * Index arithmetic for the fixed-capacity sample ring.
 */

#include <stddef.h>
#include <stdint.h>

static size_t wrap_index(size_t index, size_t capacity) {
    if (capacity == 0) {
        return 0;
    }
    while (index >= capacity) {
        index -= capacity;
    }
    return index;
}

static int buffer_is_full(size_t head, size_t tail, size_t capacity) {
    size_t used = head - tail;
    if (capacity == 0) {
        return 0;
    }
    return used >= capacity;
}

size_t locate_max_slot(const int32_t *values, size_t length) {
    size_t slot = 0;
    for (size_t k = 1; k < length; k++) {
        if (values[k] > values[slot]) {
            slot = k;
        }
    }
    return slot;
}

size_t locate_min_slot(const int32_t *values, size_t length) {
    int32_t smallest = values[0];
    size_t slot = 0;
    for (size_t k = 1; k < length; k++) {
        if (values[k] < smallest) {
            smallest = values[k];
            slot = k;
        }
    }
    return slot;
}

size_t slots_available(size_t head, size_t tail, size_t capacity) {
    size_t used = head - tail;
    if (used >= capacity) {
        return 0;
    }
    return capacity - used;
}
