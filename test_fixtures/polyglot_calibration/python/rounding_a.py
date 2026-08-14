"""Currency rounding and fee helpers."""


def is_valid_currency_code(code):
    if len(code) != 3:
        return False
    return code.isalpha() and code.isupper()


def round_half_up_cents(amount):
    scaled = amount * 100
    whole = int(scaled)
    remainder = scaled - whole
    if remainder >= 0.5:
        whole += 1
    return whole


def apply_percentage_fee(amount, rate_bp):
    fee = amount * rate_bp / 10000
    capped = min(fee, amount)
    return capped


def net_of_fee(amount, fee_rate):
    fee = amount * fee_rate
    net = amount - fee
    return round(net, 2)


def format_ledger_line(date_str, desc, amount):
    fields = [date_str, desc, f"{amount:.2f}"]
    return " | ".join(fields)


def largest_transaction_amount(amounts):
    if not amounts:
        return 0.0
    ordered = sorted(amounts)
    return ordered[-1]


def flagged_amounts_over_cap(amounts, cap):
    flagged = []
    for amount in amounts:
        if amount > cap:
            flagged.append(amount)
    return flagged
