#include "metering.h"

#include <limits.h>
#include <string.h>

static int compare_totals(const DeviceTotal *left, const DeviceTotal *right) {
    return strcmp(left->device, right->device);
}

static void sort_totals(DeviceTotal *totals, size_t count) {
    for (size_t index = 1; index < count; index++) {
        DeviceTotal current = totals[index];
        size_t position = index;
        while (position > 0 && compare_totals(&current, &totals[position - 1]) < 0) {
            totals[position] = totals[position - 1];
            position--;
        }
        totals[position] = current;
    }
}

static int validate_reading(const Reading *reading) {
    return reading != NULL && reading->device[0] != '\0'
        && memchr(reading->device, '\0', sizeof(reading->device)) != NULL
        && reading->usage >= 0;
}

static int add_usage(DeviceTotal *total, int usage) {
    if (total->total_usage > INT_MAX - usage) {
        return METERING_OVERFLOW;
    }
    total->total_usage += usage;
    return METERING_OK;
}

int aggregate_usage_linear(
    const Reading *readings,
    size_t count,
    DeviceTotal *out,
    size_t capacity,
    size_t *out_count
) {
    if (readings == NULL || out == NULL || out_count == NULL) {
        return METERING_INVALID;
    }
    *out_count = 0;
    for (size_t index = 0; index < count; index++) {
        const Reading *reading = &readings[index];
        if (!validate_reading(reading)) {
            return METERING_INVALID;
        }
        if (reading->discarded) {
            continue;
        }
        size_t slot = 0;
        while (slot < *out_count && strcmp(out[slot].device, reading->device) != 0) {
            slot++;
        }
        if (slot == *out_count) {
            if (*out_count == capacity) {
                return METERING_CAPACITY;
            }
            strcpy(out[slot].device, reading->device);
            out[slot].total_usage = 0;
            out[slot].reading_count = 0;
            (*out_count)++;
        }
        if (add_usage(&out[slot], reading->usage) != METERING_OK) {
            return METERING_OVERFLOW;
        }
        out[slot].reading_count++;
    }
    sort_totals(out, *out_count);
    return METERING_OK;
}

int aggregate_usage_sorted(
    const Reading *readings,
    size_t count,
    DeviceTotal *out,
    size_t capacity,
    size_t *out_count
) {
    size_t ordered[METERING_MAX_DEVICES];
    size_t live_count = 0;
    if (readings == NULL || out == NULL || out_count == NULL || count > METERING_MAX_DEVICES) {
        return METERING_INVALID;
    }
    *out_count = 0;
    for (size_t index = 0; index < count; index++) {
        if (!validate_reading(&readings[index])) {
            return METERING_INVALID;
        }
        if (!readings[index].discarded) {
            size_t position = live_count;
            while (position > 0 && strcmp(readings[index].device, readings[ordered[position - 1]].device) < 0) {
                ordered[position] = ordered[position - 1];
                position--;
            }
            ordered[position] = index;
            live_count++;
        }
    }
    for (size_t index = 0; index < live_count; index++) {
        const Reading *reading = &readings[ordered[index]];
        if (index == 0 || strcmp(reading->device, readings[ordered[index - 1]].device) != 0) {
            if (*out_count == capacity) {
                return METERING_CAPACITY;
            }
            strcpy(out[*out_count].device, reading->device);
            out[*out_count].total_usage = 0;
            out[*out_count].reading_count = 0;
            (*out_count)++;
        }
        if (add_usage(&out[*out_count - 1], reading->usage) != METERING_OK) {
            return METERING_OVERFLOW;
        }
        out[*out_count - 1].reading_count++;
    }
    return METERING_OK;
}

int aggregate_usage_by_index(
    const Reading *readings,
    size_t count,
    DeviceTotal *out,
    size_t capacity,
    size_t *out_count
) {
    if (readings == NULL || out == NULL || out_count == NULL) {
        return METERING_INVALID;
    }
    *out_count = 0;
    for (size_t index = 0; index < count; index++) {
        const Reading current = readings[index];
        if (!validate_reading(&current)) {
            return METERING_INVALID;
        }
        if (current.discarded) {
            continue;
        }
        int found = -1;
        for (size_t candidate = 0; candidate < *out_count; candidate++) {
            if (strcmp(out[candidate].device, current.device) == 0) {
                found = (int)candidate;
                break;
            }
        }
        if (found < 0) {
            if (*out_count >= capacity) {
                return METERING_CAPACITY;
            }
            found = (int)*out_count;
            memset(&out[found], 0, sizeof(out[found]));
            strcpy(out[found].device, current.device);
            (*out_count)++;
        }
        if (add_usage(&out[found], current.usage) != METERING_OK) {
            return METERING_OVERFLOW;
        }
        out[found].reading_count = out[found].reading_count + 1;
    }
    sort_totals(out, *out_count);
    return METERING_OK;
}
