#include "metering.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(INT_MAX >= 2147483647, "The record API requires at least 32-bit int");

MeterRecordStatus record_parse_scan(
    const unsigned char *data,
    size_t length,
    MeterRecord *output
) {
    if (data == NULL || output == NULL) {
        return METER_RECORD_BAD_ARGUMENT;
    }
    unsigned int sensor = 0U;
    unsigned int magnitude = 0U;
    size_t sensor_digits = 0U;
    size_t delta_digits = 0U;
    int field = 0;
    int negative = 0;
    int sign_seen = 0;
    int finished = 0;
    for (size_t index = 0U; index < length; ++index) {
        unsigned char byte = data[index];
        if (field == 0) {
            if (byte >= (unsigned char)'0' && byte <= (unsigned char)'9') {
                if (sensor_digits == 4U) {
                    return METER_RECORD_INVALID;
                }
                sensor = sensor * 10U + (unsigned int)(byte - (unsigned char)'0');
                ++sensor_digits;
            } else if (byte == (unsigned char)',' && sensor_digits > 0U && sensor > 0U) {
                field = 1;
            } else {
                return METER_RECORD_INVALID;
            }
        } else if (byte == (unsigned char)'-' && delta_digits == 0U && !sign_seen) {
            negative = 1;
            sign_seen = 1;
        } else if (byte >= (unsigned char)'0' && byte <= (unsigned char)'9') {
            if (delta_digits == 5U) {
                return METER_RECORD_INVALID;
            }
            magnitude = magnitude * 10U + (unsigned int)(byte - (unsigned char)'0');
            ++delta_digits;
        } else if (byte == (unsigned char)'\n' && delta_digits > 0U && index == length - 1U) {
            finished = 1;
        } else {
            return METER_RECORD_INVALID;
        }
    }
    if (!finished || magnitude > (negative ? 32768U : 32767U)) {
        return METER_RECORD_INVALID;
    }
    MeterRecord value;
    value.sensor_id = sensor;
    value.delta = negative ? -(int)magnitude : (int)magnitude;
    *output = value;
    return METER_RECORD_OK;
}

MeterRecordStatus record_parse_fields(
    const unsigned char *data,
    size_t length,
    MeterRecord *output
) {
    if (data == NULL || output == NULL) {
        return METER_RECORD_BAD_ARGUMENT;
    }
    if (length < 4U || length > 12U || data[length - 1U] != (unsigned char)'\n') {
        return METER_RECORD_INVALID;
    }
    const unsigned char *separator = memchr(data, ',', length);
    if (separator == NULL) {
        return METER_RECORD_INVALID;
    }
    size_t id_length = (size_t)(separator - data);
    size_t amount_length = length - id_length - 2U;
    if (id_length == 0U || id_length > 4U || amount_length == 0U || amount_length > 6U) {
        return METER_RECORD_INVALID;
    }
    for (size_t index = 0U; index < id_length; ++index) {
        if (data[index] < (unsigned char)'0' || data[index] > (unsigned char)'9') {
            return METER_RECORD_INVALID;
        }
    }
    const unsigned char *amount = separator + 1;
    size_t first_digit = amount[0] == (unsigned char)'-' ? 1U : 0U;
    size_t digit_count = amount_length - first_digit;
    if (digit_count == 0U || digit_count > 5U) {
        return METER_RECORD_INVALID;
    }
    for (size_t index = first_digit; index < amount_length; ++index) {
        if (amount[index] < (unsigned char)'0' || amount[index] > (unsigned char)'9') {
            return METER_RECORD_INVALID;
        }
    }
    char id_text[5] = {0};
    char amount_text[7] = {0};
    memcpy(id_text, data, id_length);
    memcpy(amount_text, amount, amount_length);
    errno = 0;
    char *end = NULL;
    long id_value = strtol(id_text, &end, 10);
    if (errno == ERANGE || *end != '\0' || id_value < 1L || id_value > 9999L) {
        return METER_RECORD_INVALID;
    }
    errno = 0;
    long amount_value = strtol(amount_text, &end, 10);
    if (errno == ERANGE || *end != '\0' || amount_value < -32768L || amount_value > 32767L) {
        return METER_RECORD_INVALID;
    }
    MeterRecord value;
    value.sensor_id = (unsigned int)id_value;
    value.delta = (int)amount_value;
    *output = value;
    return METER_RECORD_OK;
}
