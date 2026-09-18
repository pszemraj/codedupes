#ifndef METERING_H
#define METERING_H

#include <stddef.h>
#include <stdio.h>

#define METERING_MAX_DEVICES 32
#define METERING_KEY_SIZE 16
#define METERING_MAX_ERRORS 16

enum {
    METERING_OK = 0,
    METERING_INVALID = -1,
    METERING_CAPACITY = -2,
    METERING_DUPLICATE = -3,
};

typedef struct {
    char device[METERING_KEY_SIZE];
    int usage;
    int discarded;
} Reading;

typedef struct {
    unsigned int sensor_id;
    int delta;
} MeterRecord;

typedef enum {
    METER_RECORD_OK = 0,
    METER_RECORD_INVALID = 1,
    METER_RECORD_BAD_ARGUMENT = 2,
} MeterRecordStatus;

typedef struct {
    char device[METERING_KEY_SIZE];
    int total_usage;
    size_t reading_count;
} DeviceTotal;

typedef struct {
    char device[METERING_KEY_SIZE];
    int amount;
    int transfer_amount;
    int deferred_amount;
    char state[12];
} PayoutLine;

typedef struct {
    PayoutLine lines[METERING_MAX_DEVICES];
    size_t line_count;
    int transfer_count;
    int deferred_total;
    int audit_events;
} SettlementPlan;

typedef struct {
    char device[METERING_KEY_SIZE];
    char usage_text[METERING_KEY_SIZE];
} ApiReading;

typedef struct {
    Reading accepted[METERING_MAX_DEVICES];
    size_t accepted_count;
    size_t rejected_count;
    size_t first_rejected_row;
} ImportReport;

typedef struct {
    int minimum_payout;
    int defer_small_positive;
} PayoutPolicy;

int aggregate_usage_linear(
    const Reading *readings,
    size_t count,
    DeviceTotal *out,
    size_t capacity,
    size_t *out_count
);
int aggregate_usage_sorted(
    const Reading *readings,
    size_t count,
    DeviceTotal *out,
    size_t capacity,
    size_t *out_count
);
int aggregate_usage_by_index(
    const Reading *readings,
    size_t count,
    DeviceTotal *out,
    size_t capacity,
    size_t *out_count
);

int plan_payouts(const DeviceTotal *totals, size_t count, SettlementPlan *plan);
int plan_payouts_with_floor(
    const DeviceTotal *totals,
    size_t count,
    int minimum_payout,
    SettlementPlan *plan
);
int plan_payouts_with_policy(
    const DeviceTotal *totals,
    size_t count,
    PayoutPolicy policy,
    SettlementPlan *plan
);

int import_rows_inline(const char *const *rows, size_t count, ImportReport *report);
int import_rows_with_parser(const char *const *rows, size_t count, ImportReport *report);
int import_rows_two_phase(const char *const *rows, size_t count, ImportReport *report);

int import_csv_stream(FILE *input, ImportReport *report);
int import_api_batch(const ApiReading *items, size_t count, ImportReport *report);
/* seen_batch must contain an empty or previously stored NUL-terminated batch ID. */
int import_idempotent_batch(
    const char *batch_id,
    const ApiReading *items,
    size_t count,
    char seen_batch[32],
    ImportReport *report
);

MeterRecordStatus record_parse_scan(
    const unsigned char *data,
    size_t length,
    MeterRecord *output
);
MeterRecordStatus record_parse_fields(
    const unsigned char *data,
    size_t length,
    MeterRecord *output
);

#endif
