#include "metering.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static void begin_import(ImportReport *report) {
    memset(report, 0, sizeof(*report));
}

static void note_rejection(ImportReport *report, size_t row) {
    size_t rejected_before = report->rejected_count;
    report->rejected_count = rejected_before + 1;
    if (rejected_before == 0) {
        report->first_rejected_row = row;
    }
}

static int valid_device_character(unsigned char character) {
    return isupper(character) || isdigit(character) || character == '_' || character == '-';
}

static int normalize_csv_line(char *line, Reading *reading) {
    char *separator;
    char *end;
    long usage;
    size_t key_length;
    if (line == NULL || reading == NULL || (separator = strchr(line, ':')) == NULL) {
        return METERING_INVALID;
    }
    key_length = (size_t)(separator - line);
    if (key_length == 0 || key_length >= METERING_KEY_SIZE || separator[1] == '\0') {
        return METERING_INVALID;
    }
    for (size_t index = 0; index < key_length; index++) {
        if (!valid_device_character((unsigned char)line[index])) {
            return METERING_INVALID;
        }
    }
    errno = 0;
    usage = strtol(separator + 1, &end, 10);
    if (errno != 0 || *end != '\0' || usage < 0 || usage > INT_MAX) {
        return METERING_INVALID;
    }
    memset(reading, 0, sizeof(*reading));
    memcpy(reading->device, line, key_length);
    reading->usage = (int)usage;
    return METERING_OK;
}

static size_t bounded_length(const char *text, size_t limit) {
    size_t length = 0;
    while (length < limit && text[length] != '\0') {
        length++;
    }
    return length;
}

static int normalize_api_item(const ApiReading *item, Reading *reading) {
    char *end;
    long usage;
    size_t key_length;
    size_t value_length;
    if (item == NULL || reading == NULL) {
        return METERING_INVALID;
    }
    key_length = bounded_length(item->device, METERING_KEY_SIZE);
    value_length = bounded_length(item->usage_text, METERING_KEY_SIZE);
    if (key_length == 0 || key_length == METERING_KEY_SIZE || value_length == 0 || value_length == METERING_KEY_SIZE) {
        return METERING_INVALID;
    }
    for (size_t index = 0; index < key_length; index++) {
        if (!valid_device_character((unsigned char)item->device[index])) {
            return METERING_INVALID;
        }
    }
    errno = 0;
    usage = strtol(item->usage_text, &end, 10);
    if (errno != 0 || end != item->usage_text + value_length || usage < 0 || usage > INT_MAX) {
        return METERING_INVALID;
    }
    memset(reading, 0, sizeof(*reading));
    memcpy(reading->device, item->device, key_length);
    reading->usage = (int)usage;
    return METERING_OK;
}

int import_csv_stream(FILE *input, ImportReport *report) {
    char line[64];
    size_t row = 0;
    if (input == NULL || report == NULL) {
        return METERING_INVALID;
    }
    begin_import(report);
    while (fgets(line, sizeof(line), input) != NULL) {
        Reading normalized;
        char *newline;
        row++;
        newline = strchr(line, '\n');
        if (newline == NULL && !feof(input)) {
            int character;
            while ((character = fgetc(input)) != '\n' && character != EOF) {
            }
            note_rejection(report, row);
            continue;
        }
        if (newline != NULL) {
            *newline = '\0';
        }
        if (normalize_csv_line(line, &normalized) != METERING_OK) {
            note_rejection(report, row);
            continue;
        }
        if (report->accepted_count == METERING_MAX_DEVICES) {
            return METERING_CAPACITY;
        }
        report->accepted[report->accepted_count++] = normalized;
    }
    return ferror(input) ? METERING_INVALID : METERING_OK;
}

int import_api_batch(const ApiReading *items, size_t count, ImportReport *report) {
    if (items == NULL || report == NULL || count > METERING_MAX_DEVICES) {
        return METERING_INVALID;
    }
    begin_import(report);
    for (size_t index = 0; index < count; index++) {
        Reading normalized;
        if (normalize_api_item(&items[index], &normalized) != METERING_OK) {
            note_rejection(report, index + 1);
            continue;
        }
        report->accepted[report->accepted_count] = normalized;
        report->accepted_count++;
    }
    return METERING_OK;
}

int import_idempotent_batch(
    const char *batch_id,
    const ApiReading *items,
    size_t count,
    char seen_batch[32],
    ImportReport *report
) {
    size_t id_length;
    size_t seen_length;
    Reading staged[METERING_MAX_DEVICES];
    if (batch_id == NULL || items == NULL || seen_batch == NULL || report == NULL || count > METERING_MAX_DEVICES) {
        return METERING_INVALID;
    }
    id_length = bounded_length(batch_id, 32);
    if (id_length == 0 || id_length == 32) {
        return METERING_INVALID;
    }
    seen_length = bounded_length(seen_batch, 32);
    if (seen_length == 32) {
        return METERING_INVALID;
    }
    if (id_length == seen_length && memcmp(batch_id, seen_batch, id_length) == 0) {
        return METERING_DUPLICATE;
    }
    begin_import(report);
    for (size_t index = 0; index < count; index++) {
        if (normalize_api_item(&items[index], &staged[index]) != METERING_OK) {
            note_rejection(report, index + 1);
            return METERING_INVALID;
        }
    }
    for (size_t index = 0; index < count; index++) {
        report->accepted[report->accepted_count++] = staged[index];
    }
    memcpy(seen_batch, batch_id, id_length + 1);
    return METERING_OK;
}
