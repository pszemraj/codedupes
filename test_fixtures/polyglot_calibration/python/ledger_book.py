"""Ledger bookkeeping: recording entries and querying recent activity."""


class LedgerBook:
    def __init__(self, opening_balance=0.0):
        self.opening_balance = opening_balance
        self.entries = []

    def record_entry(self, amount, memo):
        if amount == 0:
            return
        self.entries.append({"amount": amount, "memo": memo})

    def latest_entry_timestamp(self, timestamps):
        latest = timestamps[0]
        for ts in timestamps[1:]:
            if ts > latest:
                latest = ts
        return latest

    def credit_entries(self):
        return [entry for entry in self.entries if entry["amount"] > 0]

    def has_reference(self, ref, index=0):
        if index >= len(self.entries):
            return False
        if self.entries[index].get("memo") == ref:
            return True
        return self.has_reference(ref, index + 1)
