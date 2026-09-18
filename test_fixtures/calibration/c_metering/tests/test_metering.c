#include "metering.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            exit(1); \
        } \
    } while (0)

static void check_same_report(const ImportReport *left, const ImportReport *right) {
    CHECK(memcmp(left, right, sizeof(*left)) == 0);
}

static void test_aggregation_differential(void) {
    const Reading readings[] = {
        {"METER_B", 4, 0},
        {"METER_A", 7, 0},
        {"METER_B", 5, 0},
        {"METER_C", 9, 1},
    };
    Reading preserved[4];
    DeviceTotal linear[METERING_MAX_DEVICES] = {0};
    DeviceTotal sorted[METERING_MAX_DEVICES] = {0};
    DeviceTotal indexed[METERING_MAX_DEVICES] = {0};
    size_t linear_count = 0;
    size_t sorted_count = 0;
    size_t indexed_count = 0;
    memcpy(preserved, readings, sizeof(readings));
    CHECK(aggregate_usage_linear(readings, 4, linear, METERING_MAX_DEVICES, &linear_count) == METERING_OK);
    CHECK(aggregate_usage_sorted(readings, 4, sorted, METERING_MAX_DEVICES, &sorted_count) == METERING_OK);
    CHECK(aggregate_usage_by_index(readings, 4, indexed, METERING_MAX_DEVICES, &indexed_count) == METERING_OK);
    CHECK(linear_count == 2 && sorted_count == 2 && indexed_count == 2);
    CHECK(memcmp(linear, sorted, sizeof(DeviceTotal) * linear_count) == 0);
    CHECK(memcmp(linear, indexed, sizeof(DeviceTotal) * linear_count) == 0);
    CHECK(strcmp(linear[0].device, "METER_A") == 0 && linear[0].total_usage == 7);
    CHECK(strcmp(linear[1].device, "METER_B") == 0 && linear[1].total_usage == 9);
    CHECK(memcmp(readings, preserved, sizeof(readings)) == 0);
}

static void test_invalid_discarded_reading_is_rejected(void) {
    const Reading readings[] = {{"", 3, 1}};
    DeviceTotal totals[METERING_MAX_DEVICES] = {0};
    size_t count = 0;
    CHECK(aggregate_usage_linear(readings, 1, totals, METERING_MAX_DEVICES, &count) == METERING_INVALID);
    CHECK(aggregate_usage_sorted(readings, 1, totals, METERING_MAX_DEVICES, &count) == METERING_INVALID);
    CHECK(aggregate_usage_by_index(readings, 1, totals, METERING_MAX_DEVICES, &count) == METERING_INVALID);
}

static void test_payout_floor_and_policy(void) {
    const DeviceTotal totals[] = {
        {"METER_A", 7, 1},
        {"METER_B", 0, 1},
        {"METER_C", -4, 1},
    };
    SettlementPlan baseline;
    SettlementPlan floor_zero;
    SettlementPlan policy_zero;
    SettlementPlan deferred;
    SettlementPlan policy_deferred;
    SettlementPlan policy_transferred;
    CHECK(plan_payouts(totals, 3, &baseline) == METERING_OK);
    CHECK(plan_payouts_with_floor(totals, 3, 0, &floor_zero) == METERING_OK);
    CHECK(plan_payouts_with_policy(totals, 3, (PayoutPolicy){0, 1}, &policy_zero) == METERING_OK);
    CHECK(memcmp(&baseline, &floor_zero, sizeof(baseline)) == 0);
    CHECK(memcmp(&baseline, &policy_zero, sizeof(baseline)) == 0);
    CHECK(plan_payouts_with_floor(totals, 3, 10, &deferred) == METERING_OK);
    CHECK(strcmp(deferred.lines[0].state, "deferred") == 0);
    CHECK(deferred.lines[0].transfer_amount == 0 && deferred.lines[0].deferred_amount == 7);
    CHECK(deferred.transfer_count == 0 && deferred.audit_events == 1 && deferred.deferred_total == 7);
    CHECK(plan_payouts_with_policy(totals, 3, (PayoutPolicy){10, 1}, &policy_deferred) == METERING_OK);
    CHECK(memcmp(&deferred, &policy_deferred, sizeof(deferred)) == 0);
    CHECK(plan_payouts_with_policy(totals, 3, (PayoutPolicy){10, 0}, &policy_transferred) == METERING_OK);
    CHECK(strcmp(policy_transferred.lines[0].state, "ready") == 0);
    CHECK(policy_transferred.lines[0].transfer_amount == 7);
    CHECK(policy_transferred.transfer_count == 1 && policy_transferred.audit_events == 0);
}

static void test_helper_extraction_differential(void) {
    const char *rows[] = {"METER_A:7", "wrong", "METER-2:03", "METER_A:-1"};
    ImportReport inline_report;
    ImportReport parser_report;
    ImportReport two_phase_report;
    CHECK(import_rows_inline(rows, 4, &inline_report) == METERING_OK);
    CHECK(import_rows_with_parser(rows, 4, &parser_report) == METERING_OK);
    CHECK(import_rows_two_phase(rows, 4, &two_phase_report) == METERING_OK);
    check_same_report(&inline_report, &parser_report);
    check_same_report(&inline_report, &two_phase_report);
    CHECK(inline_report.accepted_count == 2 && inline_report.rejected_count == 2);
    CHECK(inline_report.first_rejected_row == 2);
}

static void test_csv_api_and_idempotency(void) {
    const ApiReading items[] = {
        {"METER_A", "7"},
        {"bad key", "4"},
        {"METER-2", "03"},
    };
    const ApiReading valid[] = {{"METER_A", "7"}, {"METER-2", "03"}};
    const ApiReading invalid[] = {{"METER_A", "7"}, {"METER_B", "-3"}};
    FILE *input = tmpfile();
    FILE *overlong = tmpfile();
    ImportReport csv_report;
    ImportReport api_report;
    ImportReport idempotent_report;
    char seen_batch[32] = "";
    char invalid_seen_batch[32];
    CHECK(input != NULL && overlong != NULL);
    CHECK(fputs("METER_A:7\nbad key:4\nMETER-2:03\n", input) >= 0);
    rewind(input);
    CHECK(import_csv_stream(input, &csv_report) == METERING_OK);
    CHECK(import_api_batch(items, 3, &api_report) == METERING_OK);
    check_same_report(&csv_report, &api_report);
    CHECK(fputs("METER_A:123456789012345678901234567890123456789012345678901234567890123456789\nMETER-2:03\n", overlong) >= 0);
    rewind(overlong);
    CHECK(import_csv_stream(overlong, &csv_report) == METERING_OK);
    CHECK(csv_report.rejected_count == 1 && csv_report.first_rejected_row == 1);
    CHECK(csv_report.accepted_count == 1 && strcmp(csv_report.accepted[0].device, "METER-2") == 0);
    CHECK(import_idempotent_batch("batch-7", valid, 2, seen_batch, &idempotent_report) == METERING_OK);
    CHECK(idempotent_report.accepted_count == 2 && strcmp(seen_batch, "batch-7") == 0);
    CHECK(import_idempotent_batch("batch-7", valid, 2, seen_batch, &idempotent_report) == METERING_DUPLICATE);
    CHECK(import_idempotent_batch("batch-8", invalid, 2, seen_batch, &idempotent_report) == METERING_INVALID);
    CHECK(idempotent_report.first_rejected_row == 2 && strcmp(seen_batch, "batch-7") == 0);
    memset(invalid_seen_batch, 'x', sizeof(invalid_seen_batch));
    CHECK(import_idempotent_batch("batch-9", valid, 2, invalid_seen_batch, &idempotent_report) == METERING_INVALID);
    fclose(input);
    fclose(overlong);
}

static void compare_record_parsers(const unsigned char *data, size_t length) {
    MeterRecord scanned;
    MeterRecord sliced;
    unsigned char before[sizeof(MeterRecord)];
    memset(&scanned, 0x5A, sizeof(scanned));
    memset(&sliced, 0x5A, sizeof(sliced));
    memcpy(before, &scanned, sizeof(before));
    MeterRecordStatus scan_status = record_parse_scan(data, length, &scanned);
    MeterRecordStatus field_status = record_parse_fields(data, length, &sliced);
    CHECK(scan_status == field_status);
    if (scan_status == METER_RECORD_OK) {
        CHECK(scanned.sensor_id == sliced.sensor_id);
        CHECK(scanned.delta == sliced.delta);
    } else {
        CHECK(memcmp(before, &scanned, sizeof(before)) == 0);
        CHECK(memcmp(before, &sliced, sizeof(before)) == 0);
    }
}

static void test_bounded_record_parsers(void) {
    const char *invalid[] = {
        "", "1,0", "0,1\n", "10000,1\n", "1,32768\n", "1,-32769\n",
        "1,+1\n", "1,--1\n", "1,1-\n", "1,\n", ",1\n", "1,1\r\n",
        "1,1\nx", "1,1,2\n", " 1,2\n", "1, 2\n", "00001,2\n", "1,000001\n",
    };
    MeterRecord value = {9U, 9};
    for (size_t index = 0U; index < sizeof(invalid) / sizeof(invalid[0]); ++index) {
        const unsigned char *bytes = (const unsigned char *)invalid[index];
        size_t length = strlen(invalid[index]);
        CHECK(record_parse_scan(bytes, length, &value) == METER_RECORD_INVALID);
        CHECK(record_parse_fields(bytes, length, &value) == METER_RECORD_INVALID);
        compare_record_parsers(bytes, length);
    }
    const char *edge[] = {"1,-32768\n", "9999,32767\n", "0001,-0\n"};
    const unsigned int expected_id[] = {1U, 9999U, 1U};
    const int expected_delta[] = {-32768, 32767, 0};
    for (size_t index = 0U; index < sizeof(edge) / sizeof(edge[0]); ++index) {
        const unsigned char *bytes = (const unsigned char *)edge[index];
        size_t length = strlen(edge[index]);
        CHECK(record_parse_scan(bytes, length, &value) == METER_RECORD_OK);
        CHECK(value.sensor_id == expected_id[index] && value.delta == expected_delta[index]);
        CHECK(record_parse_fields(bytes, length, &value) == METER_RECORD_OK);
        CHECK(value.sensor_id == expected_id[index] && value.delta == expected_delta[index]);
    }
    CHECK(record_parse_scan(NULL, 0U, &value) == METER_RECORD_BAD_ARGUMENT);
    CHECK(record_parse_fields(NULL, 0U, &value) == METER_RECORD_BAD_ARGUMENT);
    CHECK(record_parse_scan((const unsigned char *)"1,1\n", 4U, NULL) == METER_RECORD_BAD_ARGUMENT);
    CHECK(record_parse_fields((const unsigned char *)"1,1\n", 4U, NULL) == METER_RECORD_BAD_ARGUMENT);
    const unsigned char nul[] = {'1', ',', '1', 0U, '\n'};
    CHECK(record_parse_scan(nul, sizeof(nul), &value) == METER_RECORD_INVALID);
    CHECK(record_parse_fields(nul, sizeof(nul), &value) == METER_RECORD_INVALID);
    for (unsigned int sensor = 1U; sensor <= 32U; ++sensor) {
        for (int delta = -32; delta <= 32; ++delta) {
            char text[32];
            int written = snprintf(text, sizeof(text), "%u,%d\n", sensor, delta);
            CHECK(written > 0 && (size_t)written < sizeof(text));
            compare_record_parsers((const unsigned char *)text, (size_t)written);
            CHECK(record_parse_scan((const unsigned char *)text, (size_t)written, &value) == METER_RECORD_OK);
            CHECK(value.sensor_id == sensor && value.delta == delta);
        }
    }
    uint32_t state = UINT32_C(20);
    for (size_t trial = 0U; trial < 20000U; ++trial) {
        unsigned char data[32];
        unsigned char before[32];
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        size_t length = (size_t)(state % UINT32_C(32));
        for (size_t index = 0U; index < length; ++index) {
            state = state * UINT32_C(1664525) + UINT32_C(1013904223);
            data[index] = (unsigned char)(state >> 24U);
        }
        memcpy(before, data, length);
        compare_record_parsers(data, length);
        CHECK(memcmp(before, data, length) == 0);
    }
}

int main(void) {
    test_aggregation_differential();
    test_invalid_discarded_reading_is_rejected();
    test_payout_floor_and_policy();
    test_helper_extraction_differential();
    test_csv_api_and_idempotency();
    test_bounded_record_parsers();
    puts("c_metering: all tests passed");
    return 0;
}
