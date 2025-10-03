# data_processor.py
"""
Data processing module for handling CSV conversion, Excel processing, 
and data standardization
"""

import logging
import pandas as pd
from io import StringIO
import re
import numpy as np
from PyQt6.QtWidgets import QMessageBox
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from config import TARGET_PATTERNS, ABBREVIATIONS, GENERALIZED_COLUMNS


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
    def preprocess_column_name(col_name):
        """
        Preprocess column name for better matching
        
        Args:
            col_name (str): Original column name
            
        Returns:
            str: Preprocessed column name
        """
        if not col_name or pd.isna(col_name):
            return ""
        
        # Convert to lowercase and strip
        processed = str(col_name).lower().strip()
        
        # Remove special characters and numbers, keep spaces
        processed = re.sub(r'[^\w\s]', ' ', processed)
        processed = re.sub(r'\d+', '', processed)
        
        # Replace multiple spaces with single space
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Expand abbreviations
        words = processed.split()
        expanded_words = []
        for word in words:
            expanded_words.append(ABBREVIATIONS.get(word, word))
        
        return ' '.join(expanded_words)
    
    @staticmethod
    def find_best_column_match(excel_columns, target_column, patterns, threshold=0.35):
        """
        Find best matching Excel column for a target column using cosine similarity
        
        Args:
            excel_columns (list): List of Excel column names
            target_column (str): Target column name we're trying to match
            patterns (list): List of pattern strings for the target column
            threshold (float): Minimum similarity threshold
            
        Returns:
            tuple: (best_match_column, similarity_score) or (None, 0) if no match
        """
        return None, 0  # Placeholder for actual implementation
    #     if not target_column or not patterns:
    #     if not excel_columns:
    #         return None, 0
        
    #     # Preprocess Excel columns
    #     processed_excel_cols = [DataProcessor.preprocess_column_name(col) for col in excel_columns]
        
    #     # Filter out empty column names
    #     valid_indices = [i for i, col in enumerate(processed_excel_cols) if col.strip()]
    #     if not valid_indices:
    #         return None, 0
        
    #     valid_excel_cols = [processed_excel_cols[i] for i in valid_indices]
    #     valid_original_cols = [excel_columns[i] for i in valid_indices]
        
    #     # Create corpus: Excel columns + target patterns
    #     corpus = valid_excel_cols + patterns
        
    #     try:
    #         # Create TF-IDF vectors
    #         vectorizer = TfidfVectorizer(
    #             ngram_range=(1, 2),  # Use unigrams and bigrams
    #             stop_words=None,     # Don't remove stop words for column names
    #             lowercase=True,
    #             token_pattern=r'\b\w+\b'
    #         )
            
    #         tfidf_matrix = vectorizer.fit_transform(corpus)
            
    #         # Calculate similarity between Excel columns and patterns
    #         excel_vectors = tfidf_matrix[:len(valid_excel_cols)]
    #         pattern_vectors = tfidf_matrix[len(valid_excel_cols):]
            
    #         # Get maximum similarity for each Excel column against all patterns
    #         similarities = cosine_similarity(excel_vectors, pattern_vectors)
    #         max_similarities = np.max(similarities, axis=1)
            
    #         # Find best match above threshold
    #         best_idx = np.argmax(max_similarities)
    #         best_score = max_similarities[best_idx]
            
    #         if best_score >= threshold:
    #             return valid_original_cols[best_idx], best_score
    #         else:
    #             return None, best_score
                
    #     except Exception as e:
    #         logging.warning(f"Error in similarity calculation for {target_column}: {e}")
    #         return None, 0
    
    @staticmethod
    def detect_data_boundaries(df):
        """
        Detect start and end rows of actual transaction data in Excel
        
        Args:
            df (pd.DataFrame): Raw DataFrame from Excel
            
        Returns:
            tuple: (start_row, end_row) indices for actual data
        """
        # Keywords that indicate actual data headers
        header_keywords = [
            'date', 'tran', 'txn', 'transaction', 'particulars', 'narration',
            'debit', 'credit', 'withdrawal', 'deposit', 'balance', 'amount',
            'description', 'details', 'reference', 'cheque', 'dr', 'cr'
        ]
        
        # Keywords that indicate summary/footer rows
        footer_keywords = [
            'total', 'sum', 'balance', 'closing', 'opening', 'summary',
            'grand total', 'sub total', 'final', 'end', 'footer', 'note',
            'statement', 'generated', 'printed', 'page', 'continued'
        ]
        
        start_row = 0
        end_row = len(df)
        
        # Find start row (look for header-like content)
        for idx, row in df.iterrows():
            # Convert row to string and check for header keywords
            row_text = ' '.join([str(val).lower().strip() for val in row if pd.notna(val)]).strip()
            
            if not row_text:
                continue
                
            # Count header keywords in this row
            header_matches = sum(1 for keyword in header_keywords if keyword in row_text)
            
            # If we find a row with multiple header keywords, this is likely our header
            if header_matches >= 2:
                start_row = idx
                logging.info(f"Detected header row at index {idx}: '{row_text[:100]}...'")
                break
        
        # Find end row (look for footer/summary content)
        # Start from bottom and work upwards
        for idx in range(len(df) - 1, start_row, -1):
            row = df.iloc[idx]
            row_text = ' '.join([str(val).lower().strip() for val in row if pd.notna(val)]).strip()
            
            if not row_text:
                continue
            
            # Check for footer keywords
            has_footer_keywords = any(keyword in row_text for keyword in footer_keywords)
            
            # Check if row has very few non-null values (likely summary)
            non_null_count = sum(1 for val in row if pd.notna(val) and str(val).strip())
            
            # If mostly empty or contains footer keywords, this might be start of footer
            if has_footer_keywords or non_null_count <= 2:
                # Look for actual transaction data above this row
                data_rows_above = 0
                for check_idx in range(max(start_row + 1, idx - 10), idx):
                    check_row = df.iloc[check_idx]
                    check_non_null = sum(1 for val in check_row if pd.notna(val) and str(val).strip())
                    if check_non_null >= 3:  # Likely transaction row
                        data_rows_above += 1
                
                if data_rows_above >= 3:  # Found substantial data above
                    end_row = idx
                    logging.info(f"Detected footer/summary starting at row {idx}: '{row_text[:100]}...'")
                    break
        
        # Ensure we have at least some rows to process
        if end_row <= start_row + 1:
            logging.warning("Could not detect clear data boundaries, using full dataset")
            start_row = 0
            end_row = len(df)
        
        logging.info(f"Data boundaries: rows {start_row} to {end_row-1} (total: {end_row - start_row} rows)")
        return start_row, end_row
    
    @staticmethod
    def extract_transaction_data(df):
        """
        Extract only the transaction data portion from Excel DataFrame
        
        Args:
            df (pd.DataFrame): Raw DataFrame from Excel
            
        Returns:
            pd.DataFrame: DataFrame containing only transaction data
        """
        start_row, end_row = DataProcessor.detect_data_boundaries(df)
        
        # Extract the data portion
        data_df = df.iloc[start_row:end_row].copy()
        
        # Use first row as headers if it looks like headers
        first_row = data_df.iloc[0]
        first_row_text = ' '.join([str(val).lower() for val in first_row if pd.notna(val)])
        
        header_keywords = ['date', 'tran', 'debit', 'credit', 'balance', 'amount', 'narration', 'description']
        if any(keyword in first_row_text for keyword in header_keywords):
            logging.info("Using first row of extracted data as column headers")
            data_df.columns = [str(val).strip() if pd.notna(val) else f"Col_{i}" for i, val in enumerate(first_row)]
            data_df = data_df.iloc[1:].reset_index(drop=True)
        
        # Remove any rows that are completely empty
        data_df = data_df.dropna(how='all').reset_index(drop=True)
        
        logging.info(f"Extracted {len(data_df)} transaction rows with columns: {list(data_df.columns)}")
        return data_df
    
    @staticmethod
    def flexible_column_mapping(df):
        """
        Flexibly map DataFrame columns to standardized format using cosine similarity
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            dict: Mapping of original columns to standardized columns
        """
        excel_columns = list(df.columns)
        column_mapping = {}
        used_columns = set()
        
        logging.info(f"Starting flexible column mapping for columns: {excel_columns}")
        
        # Find best matches for each target column
        for target_col, patterns in TARGET_PATTERNS.items():
            # Only consider unused columns
            available_columns = [col for col in excel_columns if col not in used_columns]
            
            best_match, score = DataProcessor.find_best_column_match(
                available_columns, target_col, patterns
            )
            
            if best_match:
                column_mapping[best_match] = target_col
                used_columns.add(best_match)
                logging.info(f"✅ Mapped '{best_match}' -> '{target_col}' (similarity: {score:.3f})")
            else:
                logging.warning(f"⚠️ No suitable match found for '{target_col}' (best score: {score:.3f})")
        
        # Log unmatched columns
        unmatched_columns = [col for col in excel_columns if col not in used_columns]
        for col in unmatched_columns:
            if col and str(col).strip():  # Ignore empty column headers
                logging.info(f"ℹ️ Unmatched column ignored: '{col}'")
            else:
                logging.info("ℹ️ Empty column header ignored")
        
        logging.info(f"Final column mapping: {column_mapping}")
        return column_mapping
    
    @staticmethod
    def clean_numeric_value(value):
        """
        Clean and convert numeric values from various formats
        
        Args:
            value: Raw value that might contain commas, parentheses, etc.
            
        Returns:
            float: Cleaned numeric value or 0 if conversion fails
        """
        if pd.isna(value) or value == "":
            return 0.0
        
        # Convert to string for processing
        str_value = str(value).strip()
        
        if not str_value:
            return 0.0
        
        # Handle negative values in parentheses (123.45) -> -123.45
        if str_value.startswith('(') and str_value.endswith(')'):
            str_value = '-' + str_value[1:-1]
        
        # Remove commas, spaces, and currency symbols
        cleaned = re.sub(r'[,\s₹$€£¥]', '', str_value)
        
        # Handle cases where minus sign is at the end
        if cleaned.endswith('-'):
            cleaned = '-' + cleaned[:-1]
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            logging.warning(f"Could not convert '{value}' to numeric, using 0")
            return 0.0
    
    @staticmethod
    def parse_date_flexible(date_value):
        """
        Parse date from various formats
        
        Args:
            date_value: Date value in various formats
            
        Returns:
            str: Date in DD-MM-YYYY format or empty string if parsing fails
        """
        if pd.isna(date_value) or date_value == "":
            return ""
        
        # Common date formats to try
        date_formats = [
            "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
            "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
            "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",
            "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
            "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y",
            "%d %B %Y", "%d-%B-%Y", "%d/%B/%Y",
            "%b %d, %Y", "%B %d, %Y"
        ]
        
        str_date = str(date_value).strip()
        
        for fmt in date_formats:
            try:
                parsed_date = pd.to_datetime(str_date, format=fmt, errors='raise')
                return parsed_date.strftime("%d-%m-%Y")
            except:
                continue
        
        # Try pandas automatic parsing as last resort
        try:
            parsed_date = pd.to_datetime(str_date, errors='raise')
            return parsed_date.strftime("%d-%m-%Y")
        except:
            logging.warning(f"Could not parse date '{date_value}', leaving empty")
            return ""
    
    @staticmethod
    def map_to_generalized_format(df):
        """
        Map uploaded Excel bank statement to generalized schema using flexible matching
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        logging.info(f"Processing DataFrame with {len(df)} rows and columns: {list(df.columns)}")
        
        # First extract transaction data (skip headers/footers)
        transaction_df = DataProcessor.extract_transaction_data(df)
        
        # Get flexible column mapping
        column_mapping = DataProcessor.flexible_column_mapping(transaction_df)
        
        # Apply the mapping
        df_mapped = transaction_df.rename(columns=column_mapping).copy()
        
        # Ensure all required columns exist
        for col in GENERALIZED_COLUMNS:
            if col not in df_mapped.columns:
                df_mapped[col] = ""
                logging.info(f"Added missing column '{col}' with empty values")

        # Process Date column with flexible parsing
        if "Date" in df_mapped.columns:
            logging.info("Processing Date column...")
            df_mapped["Date"] = df_mapped["Date"].apply(DataProcessor.parse_date_flexible)
            non_empty_dates = df_mapped["Date"].str.strip().ne("").sum()
            logging.info(f"Successfully parsed {non_empty_dates} out of {len(df_mapped)} dates")

        # Compute Financial Year and Month
        df_mapped["FY"] = df_mapped["Date"].apply(DataProcessor._get_financial_year)
        df_mapped["Month"] = df_mapped["Date"].apply(DataProcessor._get_month)

        # Process numeric columns with enhanced cleaning
        numeric_columns = ["Withdrawal Amount", "Deposit Amount", "Balance"]
        for col in numeric_columns:
            if col in df_mapped.columns:
                logging.info(f"Processing numeric column: {col}")
                df_mapped[col] = df_mapped[col].apply(DataProcessor.clean_numeric_value)
                non_zero_count = (df_mapped[col] != 0).sum()
                logging.info(f"Column '{col}': {non_zero_count} non-zero values out of {len(df_mapped)}")

        # Calculate Net(Cr-Dr)
        df_mapped["Net(Cr-Dr)"] = df_mapped["Deposit Amount"] - df_mapped["Withdrawal Amount"]

        # Set default values for unmapped fields
        default_fields = ["Name", "Ledger", "Group"]
        for field in default_fields:
            if field not in df_mapped.columns or df_mapped[field].isna().all():
                df_mapped[field] = "suspense"

        # Clean Narration field if it exists
        if "Narration" in df_mapped.columns:
            df_mapped["Narration"] = df_mapped["Narration"].fillna("").astype(str)
            non_empty_narration = df_mapped["Narration"].str.strip().ne("").sum()
            logging.info(f"Narration column: {non_empty_narration} non-empty values")

        # Reorder columns to match standard format
        df_mapped = df_mapped[GENERALIZED_COLUMNS]
        
        logging.info(f"Final DataFrame shape: {df_mapped.shape}")
        return df_mapped
    
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