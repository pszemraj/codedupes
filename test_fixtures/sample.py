"""Test fixture with intentional duplicates and dead code."""


# --- Pair 1: Exact structural duplicate (different names/vars) ---


def calculate_sum(items):
    """Sum up all the items."""
    total = 0
    for item in items:
        total += item
    return total


def compute_total(values):
    """Add all values together."""
    result = 0
    for value in values:
        result += value
    return result


# --- Pair 2: Semantic duplicate (same intent, different impl) ---


def find_max_element(data):
    """Find the largest element in a list."""
    if not data:
        return None
    largest = data[0]
    for item in data[1:]:
        if item > largest:
            largest = item
    return largest


def get_maximum(numbers):
    """Return the maximum value from a sequence."""
    if len(numbers) == 0:
        raise ValueError("Empty sequence")
    return sorted(numbers)[-1]


# --- Pair 3: Near duplicate (minor logic variation) ---


def validate_email(email):
    """Check if email looks valid."""
    if "@" not in email:
        return False
    parts = email.split("@")
    if len(parts) != 2:
        return False
    domain = parts[1]
    if "." not in domain:
        return False
    return True


def check_email_format(address):
    """Verify email format."""
    if "@" not in address:
        return False
    segments = address.split("@")
    if len(segments) != 2:
        return False
    host = segments[1]
    if "." not in host:
        return False
    return len(host) > 3


# --- Dead code ---


def _unused_helper():
    """This function is never called anywhere."""
    return "nobody uses me"


def _another_dead_function(x, y):
    """Also never referenced."""
    return x * y + 42


# --- Used code (call graph links these) ---


def process_data(raw):
    """Process raw data through the pipeline."""
    cleaned = _clean(raw)
    total = calculate_sum(cleaned)
    return total


def _clean(data):
    """Clean the data."""
    return [x for x in data if x is not None]


# --- Distinctly different (should NOT match each other) ---


def fibonacci(n):
    """Generate fibonacci sequence."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq


def parse_csv_line(line):
    """Parse a single CSV line respecting quotes."""
    fields = []
    current = ""
    in_quotes = False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            fields.append(current.strip())
            current = ""
        else:
            current += char
    fields.append(current.strip())
    return fields
