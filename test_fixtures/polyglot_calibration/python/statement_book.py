"""Statement bookkeeping: mirrors LedgerBook with an independent decomposition."""


class StatementBook:
    def __init__(self, opening_balance=0.0, currency="USD"):
        self.opening_balance = opening_balance
        self.currency = currency
        self.transactions = []

    def add_transaction(self, value, note):
        if value == 0:
            return
        if len(self.transactions) >= 500:
            return
        self.transactions.append({"amount": value, "memo": note})

    def newest_mark(self, marks):
        recent = marks[0]
        for mark in marks[1:]:
            if mark >= recent:
                recent = mark
        return recent

    def filter_positive_amounts(self):
        positives = []
        for transaction in self.transactions:
            if transaction["amount"] > 0:
                positives.append(transaction)
        return positives

    def includes_tag(self, tag, start=0):
        position = start
        while position < len(self.transactions):
            if self.transactions[position].get("memo") == tag:
                return True
            position += 1
        return False
