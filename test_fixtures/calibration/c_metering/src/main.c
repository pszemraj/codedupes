#include "metering.h"

#include <stdio.h>

int main(void) {
    const Reading readings[] = {
        {"METER_A", 7, 0},
        {"METER_B", 4, 0},
        {"METER_A", 3, 0},
    };
    DeviceTotal totals[METERING_MAX_DEVICES];
    size_t total_count = 0;
    if (aggregate_usage_sorted(readings, 3, totals, METERING_MAX_DEVICES, &total_count) != METERING_OK) {
        return 1;
    }
    for (size_t index = 0; index < total_count; index++) {
        printf("%s=%d (%zu readings)\n", totals[index].device, totals[index].total_usage, totals[index].reading_count);
    }
    return 0;
}
