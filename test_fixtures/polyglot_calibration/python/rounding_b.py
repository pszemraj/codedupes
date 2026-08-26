"""Currency rounding and fee helpers, mirrored: an independently phrased counterpart."""


def is_valid_currency_code(code):
    if len(code) != 3:
        return False
    return code.isalpha() and code.isupper()


def round_half_up_cents(amount):
    """Round a decimal amount to whole cents using half-up rounding."""
    # Work in integer cents to avoid float comparison surprises later.
    scaled = amount * 100
    whole = int(scaled)
    remainder = scaled - whole
    # Anything at or past the midpoint rounds up, matching accounting convention.
    if remainder >= 0.5:
        whole += 1
    return whole


def compute_charge_amount(total, basis_points):
    charge = total * basis_points / 10000
    limited = min(charge, total)
    return limited


def value_after_charge(total, rate):
    charge = total * rate
    remainder = total - charge
    return round(remainder, 3)


def render_statement_row(day_str, memo, value):
    line = day_str
    line += " | " + memo
    line += " | " + f"{value:.2f}"
    return line


def max_entry_value(values):
    if not values:
        return 0.0
    best = values[0]
    for value in values[1:]:
        if value > best:
            best = value
    return best


def _above_ceiling(value, ceiling):
    return value > ceiling


def values_beyond_ceiling(values, ceiling):
    results = []
    for value in values:
        if _above_ceiling(value, ceiling):
            results.append(value)
    return results
