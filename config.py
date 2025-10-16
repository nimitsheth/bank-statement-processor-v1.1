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
- Chq Ref: Cheq number if mentioned in the statement otherwise leave blank
- Withdrawal Amount, Deposit Amount, Balance: Use numeric values. If any of these are not present, enter either 0 or leave blank exactly as shown in the source statement. Do not omit rows with zero values.
- Net(Cr-Dr): Negative withdrawal amount for withdrawals; positive deposit amount for deposits; use 0 or leave blank if both values are missing or zero.
- Name: Extract involved party's name from narration if possible; use 'suspense' if unidentifiable.
- Ledger: Assign ledger based on narration if clear, otherwise use 'suspense'.
- Group: Assign a relevant group such as Direct Expenses, Indirect Expenses, Purchase Accounts, Finance Expenses, or Sales Accounts; use 'suspense' if not mappable.

Required Behaviors:
- The column order is such that withdrwal column comes before deposit column. In cases where the header of the columns is not there in the pdf file and you parse keep in mind that if there are two columns one empty and one with amount or vice versa, the first one is withdrwal and the second one is deposit
- Use the specified column order for every row.
- First row should necessarily be the column headers
- For missing or unidentifiable Name, Ledger, or Group, set column value as 'suspense'.
- Remove duplicate or incomplete transactions (missing Date, Amounts, or Narration). Only output complete and non-duplicate entries.
- Do not hallucinate or omit transactions except per the above guidelines.
- Output ONLY plain CSV text: absolutely no explanations, headers, markdown, or formatting identifiers.

CSV Schema Reference:
- FY: string (e.g. 2023-2024, 2024-2025)
- Month: string, number (e.g., 1, 2)
- Date: string, 'DD-MM-YYYY'
- Narration: string, single line
- Chq Ref: string, number, blank if missing
- Withdrawal Amount: decimal (e.g., 1000.00), 0 or blank if missing
- Deposit Amount: decimal (e.g., 500.00), 0 or blank if missing
- Balance: decimal (e.g., 12000.00), 0 or blank if missing
- Net(Cr-Dr): decimal (e.g., 500.00 / -1000.00), 0 or blank if not applicable
- Name: string, or 'Suspense' if not identifiable
- Ledger: string, or 'Suspense' if not identifiable
- Group: string, or 'Suspense' if not assignable

Example (omit header):
04,2023-04-07,POS 123456 ABC STORE,-2500.00,,77544.00,-2500.00,ABC STORE,suspense,Direct Expenses
04,2023-04-09,NEFT FROM XYZ CORP,,5000.00,82544.00,5000.00,XYZ CORP,NEFT,Sales Accounts

After extraction, quickly validate that all required columns are present in every row, column order is never violated, and no explanations or extraneous formatting appear in the output. If validation fails, self-correct and regenerate a compliant CSV."""

# ------------------------
# AWS Configuration
# ------------------------
AWS_PROFILE = "nimit_sheth"
AWS_REGION = "ap-south-1"

# ------------------------
# Server Configuration
# ------------------------
AUTH_SERVER_URL = "http://127.0.0.1:5000/"


# ------------------------
# COLUMN MAPPINGS
# ------------------------
# Target patterns for each column type
TARGET_PATTERNS = {
    "Date": [
        "date", "transaction date", "tran date", "txn date", "value date", 
        "posting date", "trans date", "dt", "dated", "transaction dt",
        "value dt", "posting dt", "effective date", "process date"
    ],
    "Narration": [
        "narration", "description", "particulars", "details", "remarks",
        "transaction details", "txn details", "reference", "purpose",
        "memo", "note", "comment", "remark", "transaction description"
    ],
    "Withdrawal Amount": [
        "withdrawal", "debit", "debit amount", "paid out", "dr amount", 
        "withdraw", "withdrawal amt", "debit amt", "paid", "outgoing",
        "payment", "debited", "withdrawn", "dr", "debit bal", "expense"
    ],
    "Deposit Amount": [
        "deposit", "credit", "credit amount", "received", "cr amount", 
        "credited", "deposit amt", "credit amt", "incoming", "receipt",
        "income", "cr", "credit bal", "deposited", "received amt"
    ],
    "Balance": [
        "balance", "closing balance", "running balance", "available balance", 
        "bal", "closing bal", "available bal", "current balance",
        "outstanding balance", "account balance", "total balance"
    ]
}

# Abbreviation expansion dictionary
ABBREVIATIONS = {
    "tran": "transaction", "txn": "transaction", "dt": "date",
    "amt": "amount", "bal": "balance", "dr": "debit", "cr": "credit",
    "recv": "received", "avail": "available", "curr": "current",
    "acc": "account", "trans": "transaction", "dep": "deposit",
    "with": "withdrawal", "outst": "outstanding", "clos": "closing"
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