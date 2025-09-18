# config.py
"""
Configuration file containing all constants, prompts, and style definitions
"""

import os

# ------------------------
# API CONFIGURATION
# ------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAJ-jQTIV99Xn74OEIpDagvm7eCM6SUvUs"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.2

# ------------------------
# GEMINI PROMPT
# ------------------------
GEMINI_PROMPT = """
CRITICAL INSTRUCTION: OUTPUT ONLY CSV DATA - NO EXPLANATIONS, NO PROCESSING STEPS, NO MARKDOWN. 

Begin with a concise checklist (3-7 bullets) of the major extraction steps you will follow; keep these at a conceptual level. Extract transaction data from the above provided Bank Statement data and return only plain CSV text, strictly following these requirements:

Column Order (must match exactly): FY, Month, Date, Narration, Chq Ref Withdrawal Amount, Deposit Amount, Balance, Net(Cr-Dr), Name, Ledger, Group

Handling of Transactions:
- Treat a new date in the statement as the beginning of a new transaction.
- If a line is far apart from the previous transaction but lacks a new date, include it with the prior transaction.


Extraction Details:
- FY: Current Indian financial year according to date (eg: 2023-2024, 2024-2025)
- Month: Use the numeric value for the month (e.g., 1 for January, 2 for February), based on the transaction date.
- Date: Format as DD-MM-YYYY.
- Narration: Extract narration as given. If narration is multiline or lengthy, merge lines into a single space and eliminate extra breaks.
-Chq Ref: Cheq number if mentioned in the statement otherwise leave blank
- Withdrawal Amount, Deposit Amount, Balance: Use numeric values. If any of these are not present, enter either 0 or leave blank exactly as shown in the source statement. Do not omit rows with zero values.
- Net(Cr-Dr): Negative withdrawal amount for withdrawals; positive deposit amount for deposits; use 0 or leave blank if both values are missing or zero.
- Name: Extract involved party's name from narration if possible; use 'suspense' if unidentifiable.
- Ledger: Assign ledger based on narration if clear, otherwise use 'suspense'.
- Group: Assign a relevant group such as Direct Expenses, Indirect Expenses, Purchase Accounts, Finance Expenses, or Sales Accounts; use 'suspense' if not mappable.
"""

# ------------------------
# COLUMN MAPPINGS
# ------------------------
COLUMN_MAP = {
    "date": "Date",
    "txn date": "Date",
    "value date": "Date",

    "narration": "Narration",
    "description": "Narration",
    "details": "Narration",

    "cheque": "Chq Ref",
    "chq no": "Chq Ref",
    "reference": "Chq Ref",

    "withdrawal": "Withdrawal Amount",
    "debit": "Withdrawal Amount",
    "dr": "Withdrawal Amount",

    "deposit": "Deposit Amount",
    "credit": "Deposit Amount",
    "cr": "Deposit Amount",

    "balance": "Balance",
    "closing balance": "Balance"
}

GENERALIZED_COLUMNS = [
    "FY", "Month", "Date", "Narration", "Chq Ref",
    "Withdrawal Amount", "Deposit Amount", "Balance",
    "Net(Cr-Dr)", "Name", "Ledger", "Group"
]

# ------------------------
# UI STYLES
# ------------------------
STYLE_SHEET = """
QMainWindow {
    background-color: #f8f9fa;
}

QPushButton {
    background-color: #0d6efd;
    color: white;
    border-radius: 5px;
    padding: 8px 15px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #0b5ed7;
}

QPushButton:disabled {
    background-color: #ccc;
}

QPushButton#pdf_button {
    background-color: #198754;
}

QPushButton#pdf_button:hover {
    background-color: #157347;
}

QPushButton#excel_button {
    background-color: #fd7e14;
}

QPushButton#excel_button:hover {
    background-color: #e66a0d;
}

QPushButton#image_button {
    background-color: #6c757d;
}

QPushButton#image_button:hover {
    background-color: #5c636a;
}

QLineEdit {
    padding: 5px;
    border: 1px solid #ced4da;
    border-radius: 4px;
}

QLabel {
    color: #212529;
}

QTableWidget {
    border: 1px solid #dee2e6;
    border-radius: 4px;
    background-color: white;
}

QTableWidget::item {
    padding: 5px;
}

QTableWidget::item:selected {
    background-color: #e9ecef;
}

QScrollBar:vertical {
    border: none;
    background-color: #f8f9fa;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #dee2e6;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #ced4da;
}
"""

# ------------------------
# APPLICATION SETTINGS
# ------------------------
WINDOW_TITLE = "Bank Statement Processor"
MIN_WINDOW_SIZE = (900, 600)