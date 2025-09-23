# data_processor.py
"""
Data processing module for handling CSV conversion, Excel processing, 
and data standardization
"""

import logging
import pandas as pd
from io import StringIO
from PyQt6.QtWidgets import QMessageBox
from config import COLUMN_MAP, GENERALIZED_COLUMNS


class DataProcessor:
    """Class for handling all data processing operations"""
    
    @staticmethod
    def csv_to_dataframe(csv_text):
        """
        Convert CSV text to pandas DataFrame

        Args:
            csv_text (str): Raw CSV text

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            # --- Add header trimming logic ---
            lines = csv_text.strip().splitlines()
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

            df = pd.read_csv(
                StringIO(csv_text),
                on_bad_lines=lambda x: print(f"Bad line: {x}"),
                engine='python'
            )
            return df
        except pd.errors.EmptyDataError:
            raise Exception("No data found in CSV")
        except pd.errors.ParserError as e:
            raise Exception(f"Error parsing CSV data: {str(e)}")
    
    @staticmethod
    def process_excel_file(file_path):
        """
        Process Excel file and map to generalized format
        
        Args:
            file_path (str): Path to Excel file
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            df = pd.read_excel(file_path)
            return DataProcessor.map_to_generalized_format(df)
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")
    
    @staticmethod
    def map_to_generalized_format(df):
        """
        Map uploaded Excel bank statement to generalized schema
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Normalize headers
        df = df.rename(columns={
            col: COLUMN_MAP.get(col.strip().lower(), col) 
            for col in df.columns
        })

        # Ensure all required columns exist
        for col in GENERALIZED_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # Convert Date column to DD-MM-YYYY
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%d-%m-%Y")

        # Compute Financial Year and Month
        df["FY"] = df["Date"].apply(DataProcessor._get_financial_year)
        df["Month"] = df["Date"].apply(DataProcessor._get_month)

        # Process numeric columns
        for col in ["Withdrawal Amount", "Deposit Amount", "Balance"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Calculate Net(Cr-Dr)
        df["Net(Cr-Dr)"] = df["Deposit Amount"] - df["Withdrawal Amount"]

        # Set default values for unknown fields
        df["Name"] = "suspense"
        df["Ledger"] = "suspense"
        df["Group"] = "suspense"

        # Reorder columns to match standard format
        df = df[GENERALIZED_COLUMNS]

        return df
    
    @staticmethod
    def _get_financial_year(date_str):
        """
        Calculate Indian financial year from date string
        
        Args:
            date_str (str): Date in DD-MM-YYYY format
            
        Returns:
            str: Financial year in YYYY-YYYY format
        """
        if pd.isna(date_str):
            return ""
        
        try:
            date_obj = pd.to_datetime(date_str, format="%d-%m-%Y", errors="coerce")
            if pd.isna(date_obj):
                return ""
            
            year = date_obj.year
            if date_obj.month < 4:  # Before April
                return f"{year-1}-{year}"
            else:  # April onwards
                return f"{year}-{year+1}"
        except:
            return ""
    
    @staticmethod
    def _get_month(date_str):
        """
        Extract month number from date string
        
        Args:
            date_str (str): Date in DD-MM-YYYY format
            
        Returns:
            int: Month number (1-12) or empty string if error
        """
        try:
            return pd.to_datetime(date_str, format="%d-%m-%Y").month
        except:
            return ""
    
    @staticmethod
    def save_to_excel(df, output_path):
        """
        Save DataFrame to Excel file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            output_path (str): Output file path
        """
        try:
            logging.info("Converting CSV to Excel: %s", output_path)
            df.to_excel(output_path, index=False)
            logging.info("✅ CSV successfully saved as Excel at %s", output_path)
            return True
        except Exception as e:
            logging.error("⚠️ Error converting CSV to Excel: %s", e)
            QMessageBox.critical(
                None, 
                "Conversion Error", 
                f"Error converting CSV to Excel:\n{str(e)}"
            )
            return False