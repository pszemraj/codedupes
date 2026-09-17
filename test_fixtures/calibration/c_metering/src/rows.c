#include "metering.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static void reset_report(ImportReport *report) {
    memset(report, 0, sizeof(*report));
}

static void reject_row(ImportReport *report, size_t row_number) {
    report->rejected_count++;
    if (report->first_rejected_row == 0) {
        report->first_rejected_row = row_number;
    }
}

static int parse_row(const char *text, Reading *reading) {
    const char *separator;
    char *end;
    long usage;
    size_t key_length;
    if (text == NULL || reading == NULL || (separator = strchr(text, ':')) == NULL) {
        return METERING_INVALID;
    }
    key_length = (size_t)(separator - text);
    if (key_length == 0 || key_length >= METERING_KEY_SIZE || separator[1] == '\0') {
        return METERING_INVALID;
    }
    for (size_t index = 0; index < key_length; index++) {
        unsigned char character = (unsigned char)text[index];
        if (!isupper(character) && !isdigit(character) && character != '_' && character != '-') {
            return METERING_INVALID;
        }
    }
    errno = 0;
    usage = strtol(separator + 1, &end, 10);
    if (errno != 0 || *end != '\0' || usage < 0 || usage > INT_MAX) {
        return METERING_INVALID;
    }
    memset(reading, 0, sizeof(*reading));
    memcpy(reading->device, text, key_length);
    reading->usage = (int)usage;
    return METERING_OK;
}

static void note_inline_rejection(ImportReport *report, size_t row) {
    size_t rejected_before = report->rejected_count;
    report->rejected_count = rejected_before + 1;
    if (rejected_before == 0) {
        report->first_rejected_row = row;
    }
}

int import_rows_inline(const char *const *rows, size_t count, ImportReport *report) {
    if (rows == NULL || report == NULL || count > METERING_MAX_DEVICES) {
        return METERING_INVALID;
    }
    reset_report(report);
    for (size_t row = 0; row < count; row++) {
        const char *separator = rows[row] == NULL ? NULL : strchr(rows[row], ':');
        char *end = NULL;
        long usage = 0;
        size_t key_length = separator == NULL ? 0 : (size_t)(separator - rows[row]);
        int valid = separator != NULL && key_length > 0 && key_length < METERING_KEY_SIZE;
        for (size_t index = 0; valid && index < key_length; index++) {
            unsigned char character = (unsigned char)rows[row][index];
            valid = isupper(character) || isdigit(character) || character == '_' || character == '-';
        }
        if (valid && separator[1] != '\0') {
            errno = 0;
            usage = strtol(separator + 1, &end, 10);
            valid = errno == 0 && *end == '\0' && usage >= 0 && usage <= INT_MAX;
        } else {
            valid = 0;
        }
        if (!valid) {
            note_inline_rejection(report, row + 1);
            continue;
        }
        Reading *accepted = &report->accepted[report->accepted_count++];
        memset(accepted, 0, sizeof(*accepted));
        memcpy(accepted->device, rows[row], key_length);
        accepted->usage = (int)usage;
    }
    return METERING_OK;
}

int import_rows_with_parser(const char *const *rows, size_t count, ImportReport *report) {
    if (rows == NULL || report == NULL || count > METERING_MAX_DEVICES) {
        return METERING_INVALID;
    }
    reset_report(report);
    for (size_t row = 0; row < count; row++) {
        Reading parsed;
        if (parse_row(rows[row], &parsed) != METERING_OK) {
            reject_row(report, row + 1);
            continue;
        }
        report->accepted[report->accepted_count] = parsed;
        report->accepted_count += 1;
    }
    return METERING_OK;
}

int import_rows_two_phase(const char *const *rows, size_t count, ImportReport *report) {
    Reading decoded[METERING_MAX_DEVICES];
    int accepted[METERING_MAX_DEVICES] = {0};
    if (rows == NULL || report == NULL || count > METERING_MAX_DEVICES) {
        return METERING_INVALID;
    }
    reset_report(report);
    for (size_t row = 0; row < count; row++) {
        accepted[row] = parse_row(rows[row], &decoded[row]) == METERING_OK;
        if (!accepted[row]) {
            reject_row(report, row + 1);
        }
    }
    for (size_t row = 0; row < count; row++) {
        if (accepted[row]) {
            report->accepted[report->accepted_count] = decoded[row];
            report->accepted_count++;
        }
    }
    return METERING_OK;
}
