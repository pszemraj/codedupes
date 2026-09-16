#include "metering.h"

#include <string.h>

static int validate_totals(const DeviceTotal *totals, size_t count, SettlementPlan *plan) {
    if (totals == NULL || plan == NULL || count > METERING_MAX_DEVICES) {
        return METERING_INVALID;
    }
    for (size_t index = 0; index < count; index++) {
        if (totals[index].device[0] == '\0' || totals[index].reading_count == 0) {
            return METERING_INVALID;
        }
    }
    return METERING_OK;
}

int plan_payouts(const DeviceTotal *totals, size_t count, SettlementPlan *plan) {
    if (validate_totals(totals, count, plan) != METERING_OK) {
        return METERING_INVALID;
    }
    memset(plan, 0, sizeof(*plan));
    for (size_t index = 0; index < count; index++) {
        const DeviceTotal *total = &totals[index];
        PayoutLine *line = &plan->lines[index];
        strcpy(line->device, total->device);
        line->amount = total->total_usage;
        if (total->total_usage < 0) {
            strcpy(line->state, "refund_due");
        } else if (total->total_usage == 0) {
            strcpy(line->state, "balanced");
        } else {
            strcpy(line->state, "ready");
            line->transfer_amount = total->total_usage;
            plan->transfer_count++;
        }
        plan->line_count++;
    }
    return METERING_OK;
}

int plan_payouts_with_floor(
    const DeviceTotal *totals,
    size_t count,
    int minimum_payout,
    SettlementPlan *plan
) {
    if (validate_totals(totals, count, plan) != METERING_OK || minimum_payout < 0) {
        return METERING_INVALID;
    }
    memset(plan, 0, sizeof(*plan));
    for (size_t index = 0; index < count; index++) {
        int amount = totals[index].total_usage;
        PayoutLine line = {0};
        strcpy(line.device, totals[index].device);
        line.amount = amount;
        if (amount > 0 && amount < minimum_payout) {
            strcpy(line.state, "deferred");
            line.deferred_amount = amount;
            plan->deferred_total += amount;
            plan->audit_events++;
        } else if (amount > 0) {
            strcpy(line.state, "ready");
            line.transfer_amount = amount;
            plan->transfer_count += 1;
        } else if (amount < 0) {
            strcpy(line.state, "refund_due");
        } else {
            strcpy(line.state, "balanced");
        }
        plan->lines[plan->line_count++] = line;
    }
    return METERING_OK;
}

int plan_payouts_with_policy(
    const DeviceTotal *totals,
    size_t count,
    PayoutPolicy policy,
    SettlementPlan *plan
) {
    if (validate_totals(totals, count, plan) != METERING_OK || policy.minimum_payout < 0) {
        return METERING_INVALID;
    }
    memset(plan, 0, sizeof(*plan));
    for (size_t index = 0; index < count; index++) {
        PayoutLine *line = &plan->lines[plan->line_count];
        int amount = totals[index].total_usage;
        strcpy(line->device, totals[index].device);
        line->amount = amount;
        if (amount <= 0) {
            strcpy(line->state, amount == 0 ? "balanced" : "refund_due");
        } else if (policy.defer_small_positive && amount < policy.minimum_payout) {
            strcpy(line->state, "deferred");
            line->deferred_amount = amount;
            plan->deferred_total += amount;
            plan->audit_events += 1;
        } else {
            strcpy(line->state, "ready");
            line->transfer_amount = amount;
            plan->transfer_count += 1;
        }
        plan->line_count += 1;
    }
    return METERING_OK;
}
