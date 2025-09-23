import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import google.generativeai as genai
import pandas as pd
from io import StringIO
import logging
import threading

# ------------------------
# LOGGING SETUP
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # print to terminal
)

# ------------------------
# PROMPT
# ------------------------
prompt = """
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

After extraction, quickly validate that all required columns are present in every row, column order is never violated, and no explanations or extraneous formatting appear in the output. If validation fails, self-correct and regenerate a compliant CSV.
"""

# ------------------------
# CSV -> EXCEL FUNCTION
# ------------------------
def csv_to_excel(df, output_path="gemini_output.xlsx"):
    """Convert CSV text (from Gemini) into an Excel file."""
    try:
        logging.info("Converting CSV to Excel: %s", output_path)
        df.to_excel(output_path, index=False)
        logging.info("✅ CSV successfully saved as Excel at %s", output_path)
        return df
    except Exception as e:
        logging.error("⚠️ Error converting CSV to Excel: %s", e)
        messagebox.showerror("Conversion Error", f"Error converting CSV to Excel:\n{e}")
        return None

# ------------------------
# GEMINI FUNCTION
# ------------------------
def send_to_gemini(file_paths, prompt_text, model_name="gemini-2.5-flash", temperature=0.2):
    """
    Accepts either:
      - a single file path (str), or
      - a list/tuple of file paths
    Uploads each file to Gemini, then calls model.generate_content with
    the uploaded file objects followed by the prompt_text.
    Returns response.text (string).
    """
    logging.info("Preparing to send to Gemini: %s", file_paths)

    api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyAJ-jQTIV99Xn74OEIpDagvm7eCM6SUvUs"
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")

    genai.configure(api_key=api_key, transport="rest")

    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": temperature,
            "top_p": 1,
            "top_k": 40
        }
    )

    # upload one or many
    if isinstance(file_paths, (list, tuple)):
        uploaded_objs = []
        for p in file_paths:
            logging.info("Uploading file to Gemini: %s", p)
            uploaded = genai.upload_file(p)
            uploaded_objs.append(uploaded)
        payload = uploaded_objs + [prompt_text]
    else:
        logging.info("Uploading single file to Gemini: %s", file_paths)
        uploaded = genai.upload_file(file_paths)
        payload = [uploaded, prompt_text]

    logging.info("Calling Gemini model.generate_content with %d payload items", len(payload))
    response = model.generate_content(payload)
    logging.info("Gemini response received")
    return response.text

# ------------------------
# Column mapping dictionary (bank statement variations → generalized)
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
# Standardization function
# ------------------------
def map_to_generalized_format(df):
    """Map uploaded Excel bank statement to generalized schema."""
    # normalize headers
    df = df.rename(columns={col: COLUMN_MAP.get(col.strip().lower(), col) for col in df.columns})

    # Ensure all required columns exist
    for col in GENERALIZED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Convert Date column to DD-MM-YYYY
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%d-%m-%Y")

    # Compute FY & Month
    def get_fy(d):
        if pd.isna(d):
            return ""
        d = pd.to_datetime(d, format="%d-%m-%Y", errors="coerce")
        if pd.isna(d):
            return ""
        year = d.year
        if d.month < 4:
            return f"{year-1}-{year}"
        else:
            return f"{year}-{year+1}"

    df["FY"] = df["Date"].apply(get_fy)

    def get_month(d):
        try:
            return pd.to_datetime(d, format="%d-%m-%Y").month
        except:
            return ""
    df["Month"] = df["Date"].apply(get_month)

    # Numeric columns
    for col in ["Withdrawal Amount", "Deposit Amount", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Net(Cr-Dr)
    df["Net(Cr-Dr)"] = df["Deposit Amount"] - df["Withdrawal Amount"]

    # Fill suspense
    df["Name"] = "suspense"
    df["Ledger"] = "suspense"
    df["Group"] = "suspense"

    # Reorder columns
    df = df[GENERALIZED_COLUMNS]

    return df


# ------------------------
# LOADER CLASS
# ------------------------
# class Loader:
#     def __init__(self, parent, size=80):
#         self.parent = parent
#         self.size = size
#         self.canvas = tk.Canvas(parent, width=size, height=size, bg="#f8f9fa", highlightthickness=0)
#         self.arc = self.canvas.create_arc(
#             10, 10, size-10, size-10,
#             start=0, extent=90,
#             style="arc", outline="#0d6efd", width=5
#         )
#         self.angle = 0
#         self.running = False

#     def start(self):
#         self.running = True
#         self.canvas.place(relx=0.5, rely=0.5, anchor="center")  # center in parent
#         self.rotate()

#     def rotate(self):
#         if not self.running:
#             return
#         self.angle = (self.angle + 10) % 360
#         self.canvas.itemconfig(self.arc, start=self.angle)
#         self.parent.after(50, self.rotate)  # speed control

#     def stop(self):
#         self.running = False
#         self.canvas.place_forget()

class Loader:
    def __init__(self, parent, size=80):
        self.parent = parent
        self.size = size

        # Spinner canvas
        self.canvas = tk.Canvas(parent, width=size, height=size, bg="#f8f9fa", highlightthickness=0)
        self.arc = self.canvas.create_arc(
            10, 10, size-10, size-10,
            start=0, extent=90,
            style="arc", outline="#0d6efd", width=5
        )

        # Text label under spinner
        self.label = tk.Label(
            parent,
            text="Processing file(s)...",
            font=("Arial", 10, "bold"),
            bg="#fefeff",
            fg="black"
        )

        self.angle = 0
        self.running = False

    def start(self):
        self.running = True
        self.canvas.place(relx=0.5, rely=0.4, anchor="center")
        self.label.place(relx=0.5, rely=0.55, anchor="center")
        self.rotate()

    def rotate(self):
        if not self.running:
            return
        self.angle = (self.angle - 10) % 360
        self.canvas.itemconfig(self.arc, start=self.angle)
        self.parent.after(50, self.rotate)

    def stop(self):
        self.running = False
        self.canvas.place_forget()
        self.label.place_forget()   # 👈 hides the text as well


# ------------------------
# TKINTER APP
# ------------------------
class GeminiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bank Statement Processor")
        self.root.geometry("900x600")
        self.root.configure(bg="#f8f9fa")

        self.file_path = None

        # --- Top Frame: Filename + Process button ---
        top_frame = tk.Frame(root, bg="#f8f9fa")
        top_frame.pack(pady=10, padx=10, fill="x")

        tk.Label(top_frame, text="Excel File Name:", bg="#f8f9fa").pack(side="left", padx=(0, 5))
        self.filename_entry = tk.Entry(top_frame, width=40)
        self.filename_entry.pack(side="left", padx=(0, 10))

        self.process_button = tk.Button(
            top_frame,
            text="Process",
            command=self.process_file,
            state="disabled",
            bg="#0d6efd",
            fg="white"
        )
        self.process_button.pack(side="left")

        self.convert_button = tk.Button(
            top_frame,
            text="Convert to Excel",
            command=self.save_to_excel,
            state="disabled",
            bg="#0d6efd",
            fg="white"
        )
        self.convert_button.pack(side="left", padx=10)

        # --- File Info Label ---
        self.file_label = tk.Label(
            root,
            text="No file uploaded",
            fg="gray",
            bg="#f8f9fa",
            font=("Arial", 10, "italic")
        )
        self.file_label.pack(pady=5)

        table_frame = tk.Frame(root, bg="#f8f9fa")
        table_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.tree = ttk.Treeview(table_frame, show="headings")
        self.tree.pack(side="left", expand=True, fill="both")

        # Add vertical scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")

        self.tree.configure(yscrollcommand=scrollbar.set)

        self.loader = Loader(table_frame)

        # --- Bottom Frame: Upload Buttons ---
        bottom_frame = tk.Frame(root, bg="#f8f9fa")
        bottom_frame.pack(pady=15)

        self.pdf_button = tk.Button(
            bottom_frame,
            text="Upload PDF",
            command=self.upload_pdf,
            width=15,
            bg="#198754",
            fg="white"
        )
        self.pdf_button.pack(side="left", padx=20)

        self.excel_button = tk.Button(
            bottom_frame,
            text="Upload Excel",
            command=self.upload_excel,
            width=15,
            bg="#fd7e14",  # orange button
            fg="white"
        )
        self.excel_button.pack(side="left", padx=20)


        self.image_button = tk.Button(
            bottom_frame,
            text="Upload Image",
            command=self.upload_image,
            width=15,
            bg="#6c757d",
            fg="white"
        )
        self.image_button.pack(side="left", padx=20)

    # --- File Upload Handlers ---
    
    # ------------------------
    # Upload handlers (make states explicit)
    # ------------------------
    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            # clear any previous multi-image state
            if hasattr(self, "file_paths"):
                delattr = False
                try:
                    del self.file_paths
                except Exception:
                    pass

            self.file_path = file_path
            self.file_label.config(text=f"Uploaded PDF: {os.path.basename(file_path)}", fg="black")
            self.process_button.config(state="normal")
            logging.info("PDF uploaded: %s", file_path)


    def upload_image(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp;*.tiff")]
        )
        if file_paths:
            # store list of files and clear single-file state
            self.file_paths = list(file_paths)
            try:
                del self.file_path
            except Exception:
                pass

            if len(file_paths) == 1:
                label_text = f"Uploaded Image: {os.path.basename(file_paths[0])}"
            else:
                label_text = f"{len(file_paths)} Images uploaded (e.g., {os.path.basename(file_paths[0])} ...)"
            self.file_label.config(text=label_text, fg="black")
            self.process_button.config(state="normal")
            logging.info("Images uploaded: %s", file_paths)

    def upload_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"Uploaded Excel: {os.path.basename(file_path)}", fg="black")
            self.process_button.config(state="normal")
            logging.info("Excel uploaded: %s", file_path)


    # ------------------------
    # Process File (handles single PDF or multiple images)
    # ------------------------
    def process_file(self):
        # determine input source
        file_input = None
        input_type = None  # "pdf", "images", "excel"

        if hasattr(self, "file_paths") and getattr(self, "file_paths"):
            file_input = self.file_paths
            input_type = "images"
        elif hasattr(self, "file_path") and getattr(self, "file_path"):
            if self.file_path.lower().endswith((".xlsx", ".xls")):
                input_type = "excel"
                file_input = self.file_path
            else:
                input_type = "pdf"
                file_input = self.file_path
        else:
            messagebox.showwarning("No File", "Please upload a PDF, Excel, or image(s) first.")
            logging.warning("Process attempted without file(s)")
            return

        # confirm with user
        if input_type == "images":
            proceed = messagebox.askokcancel(
                "Processing", f"Are you sure you want to process {len(file_input)} image(s)?"
            )
        else:
            proceed = messagebox.askokcancel(
                "Processing", f"Are you sure you want to process the file {os.path.basename(file_input)}?"
            )

        if not proceed:
            logging.info("Processing cancelled by user")
            return

        try:
            if input_type == "excel":
                # ------------------------
                # Handle Excel locally
                # ------------------------
                logging.info("Processing Excel file locally: %s", file_input)
                self.df = pd.read_excel(file_input)
                self.df = map_to_generalized_format(self.df)
                self.show_dataframe(self.df)
                logging.info("Excel file standardized. Rows=%d", len(self.df))
                self.process_button.config(state="disabled")
                self.convert_button.config(state="normal")
                return  # stop here, don't call Gemini

            # ------------------------
            # Handle PDF / Images via Gemini (background thread)
            # ------------------------
            def task():
                try:
                    logging.info("Sending file(s) to Gemini for processing...")
                    self.loader.start()
                    self.answer = send_to_gemini(file_input, prompt)
                    self.loader.stop()

                    # parse CSV safely
                    csv_text = self.answer.strip()
                    lines = csv_text.splitlines()
                    header_idx = None
                    for i, ln in enumerate(lines):
                        if ln.strip().startswith("FY") or "FY" in ln.split(",")[0]:
                            header_idx = i
                            break
                        if ln.strip().lower().startswith("fy"):
                            header_idx = i
                            break

                    if header_idx is not None and header_idx > 0:
                        logging.info("Detected non-csv preface; trimming lines before header (line %d)", header_idx)
                        csv_text = "\n".join(lines[header_idx:])

                    try:
                        self.df = pd.read_csv(StringIO(csv_text))
                    except Exception as parse_err:
                        logging.error("pd.read_csv failed: %s", parse_err)
                        candidate = None
                        for i in range(len(lines)):
                            if "," in lines[i] and lines[i].count(",") >= 6:
                                candidate = "\n".join(lines[i:])
                                break
                        if candidate:
                            self.df = pd.read_csv(StringIO(candidate))
                        else:
                            raise parse_err

                    # update UI safely from main thread
                    self.root.after(0, lambda: [
                        self.show_dataframe(self.df),
                        logging.info("Processing complete. DataFrame rows=%d", len(self.df)),
                        self.convert_button.config(state="normal"),
                    ])

                except Exception as e:
                    self.loader.stop()
                    logging.exception("Error processing file(s)")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing file(s):\n{e}"))
                finally:
                    self.root.after(0, lambda: [self.file_label.config(text="Processing completed", fg="black"),
                                                self.process_button.config(state="disabled")])
            threading.Thread(target=task, daemon=True).start()

        except Exception as e:
            self.loader.stop()
            logging.exception("Error starting processing thread")
            messagebox.showerror("Error", f"Error starting processing:\n{e}")        



    # --- Show DataFrame in Treeview ---
    def show_dataframe(self, df):
        logging.info("Displaying DataFrame preview in UI")

        # Clear old data
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)

        # Configure columns
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=100)

        # Insert rows
        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))
    
    def save_to_excel(self):
        if hasattr(self, "df") and self.df is not None:
            output_name = self.filename_entry.get().strip()
            if not output_name:
                output_name = self.file_path
            if not output_name.endswith(".xlsx"):
                output_name += ".xlsx"
            csv_to_excel(self.df, output_name)
            messagebox.showinfo("Success", f"Excel saved as {output_name}")
            self.tree.delete(*self.tree.get_children())
            self.filename_entry.delete(0, tk.END)
            self.file_label.config(text="No file uploaded", fg="black")
            self.convert_button.config(state="disabled")
            self.file_label.config(text="Sucessfuly converted to Excel. Please upload the next file for processing.", fg="black")
        else:
            messagebox.showwarning("No Data", "No processed data to save.")
        
        


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    logging.info("Starting Bank Statement Processor App")
    root = tk.Tk()
    app = GeminiApp(root)
    root.mainloop()
    logging.info("App closed")
