# data_processor.py
"""
Data processing module for handling CSV conversion, Excel processing, 
and data standardization
"""

import logging
import csv
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
    
    # @staticmethod
    # def csv_to_dataframe(csv_text):
    #     """
    #     Convert CSV text (possibly containing multiple 'Table:' sections)
    #     into a single pandas DataFrame.

    #     Header detection strategy:
    #       - In each "Table:" section, look at the first few non-empty lines (e.g. 6).
    #       - Use csv.reader to split lines (handles quoted commas).
    #       - Pick the first line with >2 tokens and where tokens contain no digits (primary rule).
    #       - If not found, fallback to: uppercase-all-tokens heuristic OR reuse last seen header.
    #     """

    #     if not csv_text or not csv_text.strip():
    #         raise Exception("Empty CSV text")

    #     # Split into Table: sections
    #     lines = csv_text.splitlines()
    #     sections = []
    #     current = []
    #     for ln in lines:
    #         if ln.strip().startswith("Table:"):
    #             if current:
    #                 sections.append(current)
    #             current = [ln]
    #         else:
    #             current.append(ln)
    #     if current:
    #         sections.append(current)

    #     dfs = []
    #     last_header = None

    #     def tokens_for_line(line):
    #         # csv.reader returns list of tokens respecting quotes
    #         try:
    #             return next(csv.reader([line]))
    #         except Exception:
    #             # fallback naive split
    #             return [t.strip() for t in line.split(",")]

    #     for sec in sections:
    #         # remove leading Table: lines and blanks
    #         sec_lines = [l for l in sec if not l.strip().startswith("Table:")]
    #         sec_lines = [l for l in sec_lines if l.strip()]
    #         if not sec_lines:
    #             continue

    #         # Consider only first N candidate lines for header detection
    #         N = min(6, len(sec_lines))
    #         header_idx = None

    #         for i in range(N):
    #             line = sec_lines[i].strip()
    #             toks = tokens_for_line(line)
    #             # require >2 tokens (at least 3 columns)
    #             if len(toks) <= 2:
    #                 continue

    #             # primary rule: none of tokens contain a digit
    #             if not any(re.search(r'\d', t) for t in toks):
    #                 header_idx = i
    #                 break

    #             # secondary: tokens mostly alphabetic (allow short numeric tokens like "NO")
    #             alpha_count = sum(1 for t in toks if re.match(r'^[A-Za-z\.\s]+$', t.strip()))
    #             if alpha_count >= max(2, len(toks) - 1):
    #                 header_idx = i
    #                 break

    #         # fallback uppercase-all-tokens heuristic (common in OCR headers)
    #         if header_idx is None:
    #             for i in range(N):
    #                 toks = tokens_for_line(sec_lines[i])
    #                 if len(toks) > 2 and all(any(c.isalpha() for c in t) and t.strip() == t.strip().upper() for t in toks if t.strip()):
    #                     header_idx = i
    #                     break

    #         # Use last header when header not found
    #         if header_idx is None:
    #             if last_header is None:
    #                 # nothing to do for this section
    #                 continue
    #             header = last_header
    #             data_lines = sec_lines
    #         else:
    #             header = sec_lines[header_idx].strip()
    #             last_header = header
    #             data_lines = sec_lines[header_idx + 1 :]

    #         # filter out empty and summary/footer lines
    #         filtered = []
    #         for l in data_lines:
    #             s = l.strip()
    #             if not s:
    #                 continue
    #             low = s.lower()
    #             if low.startswith("page total") or low.startswith("page") or low.startswith("total") or "page total" in low:
    #                 continue
    #             filtered.append(l)

    #         if not filtered:
    #             continue

    #         csv_chunk = "\n".join([header] + filtered)
    #         # parse with pandas, fallback to manual normalization
    #         try:
    #             df_chunk = pd.read_csv(StringIO(csv_chunk), engine="python")
    #         except Exception:
    #             reader = csv.reader(StringIO(csv_chunk))
    #             rows = list(reader)
    #             if not rows:
    #                 continue
    #             header_row = [h.strip() for h in rows[0]]
    #             ncols = len(header_row)
    #             normalized = []
    #             for r in rows[1:]:
    #                 if len(r) < ncols:
    #                     r = r + [""] * (ncols - len(r))
    #                 elif len(r) > ncols:
    #                     r = r[: ncols - 1] + [",".join(r[ncols - 1 :])]
    #                 normalized.append([c.strip() for c in r])
    #             df_chunk = pd.DataFrame(normalized, columns=header_row)

    #         dfs.append(df_chunk)

    #     if not dfs:
    #         raise Exception("No table data found in CSV text")

    #     df_all = pd.concat(dfs, ignore_index=True, sort=False).fillna("")
    #     df_all.columns = [str(c).strip() for c in df_all.columns]
    #     return df_all


    @staticmethod
    def csv_to_dataframe(csv_text):
        """
        Convert CSV text (possibly containing multiple 'Table:' sections)
        into a single pandas DataFrame with smart column alignment.

        Header detection strategy:
        - In each "Table:" section, look at the first few non-empty lines (e.g. 6).
        - Use csv.reader to split lines (handles quoted commas).
        - Pick the first line with >2 tokens and where tokens contain no digits (primary rule).
        - If not found, fallback to: uppercase-all-tokens heuristic OR reuse last seen header.
        - Handle continuation tables with fewer columns by analyzing data patterns.
        """

        if not csv_text or not csv_text.strip():
            raise Exception("Empty CSV text")

        # Split into Table: sections
        lines = csv_text.splitlines()
        sections = []
        current = []
        for ln in lines:
            if ln.strip().startswith("Table:"):
                if current:
                    sections.append(current)
                current = [ln]
            else:
                current.append(ln)
        if current:
            sections.append(current)

        dfs = []
        last_header = None
        last_column_count = 0

        def tokens_for_line(line):
            # csv.reader returns list of tokens respecting quotes
            try:
                return next(csv.reader([line]))
            except Exception:
                # fallback naive split
                return [t.strip() for t in line.split(",")]

        def detect_pattern(value):
            """
            Detect the pattern type of a value.
            Returns: 'date', 'amount', 'text', or 'empty'
            """
            if not value or pd.isna(value):
                return 'empty'
            
            val_str = str(value).strip()
            if not val_str:
                return 'empty'
            
            # Date patterns: DD-MM-YY, DD/MM/YY, DD.MM.YY, etc.
            date_pattern = r'^\d{1,2}[-/.]\d{1,2}[-/.](\d{2}|\d{4})$'
            if re.match(date_pattern, val_str):
                return 'date'
            
            # Amount patterns: quoted or unquoted numbers with commas, decimals, Cr/Dr
            # Remove quotes for checking
            unquoted = val_str.strip('"').strip("'")
            
            # Check for amount indicators
            has_comma = ',' in unquoted
            has_decimal = '.' in unquoted
            has_cr_dr = unquoted.endswith('Cr') or unquoted.endswith('Dr')
            has_digits = bool(re.search(r'\d', unquoted))
            has_currency = any(sym in unquoted for sym in ['₹', '$', '€', '£', '¥'])
            in_parentheses = unquoted.startswith('(') and unquoted.endswith(')')
            
            # Amount if: has digits AND (has comma OR has decimal OR has Cr/Dr OR has currency OR in parentheses)
            if has_digits and (has_comma or has_decimal or has_cr_dr or has_currency or in_parentheses):
                return 'amount'
            
            # If mostly digits (>50% of non-space characters are digits)
            digit_count = sum(c.isdigit() for c in unquoted)
            total_chars = len(unquoted.replace(' ', ''))
            if total_chars > 0 and digit_count / total_chars > 0.5:
                return 'amount'
            
            # Otherwise it's text
            return 'text'

        def analyze_column_patterns(data_rows, num_samples=10):
            """
            Analyze data patterns for each column position.
            Returns list of pattern types for each column.
            """
            if not data_rows:
                return []
            
            # Sample first N rows
            sample_rows = data_rows[:min(num_samples, len(data_rows))]
            num_cols = max(len(row) for row in sample_rows) if sample_rows else 0
            
            column_patterns = []
            for col_idx in range(num_cols):
                patterns = []
                empty_count = 0
                
                for row in sample_rows:
                    if col_idx < len(row):
                        pattern = detect_pattern(row[col_idx])
                        patterns.append(pattern)
                        if pattern == 'empty':
                            empty_count += 1
                    else:
                        patterns.append('empty')
                        empty_count += 1
                
                # Determine dominant pattern
                if empty_count >= len(sample_rows) * 0.8:  # 80% empty
                    column_patterns.append('mostly_empty')
                else:
                    # Get most common non-empty pattern
                    non_empty = [p for p in patterns if p != 'empty']
                    if non_empty:
                        # Count pattern frequencies
                        pattern_counts = {}
                        for p in non_empty:
                            pattern_counts[p] = pattern_counts.get(p, 0) + 1
                        dominant = max(pattern_counts, key=pattern_counts.get)
                        column_patterns.append(dominant)
                    else:
                        column_patterns.append('empty')
            
            return column_patterns

        def get_reference_patterns(header_tokens):
            """
            Get expected patterns based on header column names.
            """
            patterns = []
            for token in header_tokens:
                token_lower = token.lower().strip()
                
                # Date columns
                if any(kw in token_lower for kw in ['date', 'dt', 'txn date', 'transaction date', 'value date']):
                    patterns.append('date')
                # Amount columns
                elif any(kw in token_lower for kw in ['amount', 'withdrawal', 'deposit', 'balance', 'debit', 'credit', 'dr', 'cr']):
                    patterns.append('amount')
                # Cheque/reference number (often empty)
                elif any(kw in token_lower for kw in ['chq', 'cheque', 'ref', 'reference', 'check']):
                    patterns.append('mostly_empty')
                # Text columns
                else:
                    patterns.append('text')
            
            return patterns

        def align_columns(data_rows, current_patterns, reference_header, reference_patterns):
            """
            Align data columns to match reference header when column count differs.
            Returns aligned data rows with empty columns inserted where needed.
            """
            if len(current_patterns) >= len(reference_patterns):
                # No alignment needed
                return data_rows
            
            # Find which column is missing by comparing patterns
            aligned_rows = []
            missing_position = None
            
            # Try to find where the mismatch occurs
            for ref_idx in range(len(reference_patterns)):
                if ref_idx >= len(current_patterns):
                    missing_position = ref_idx
                    break
                
                if reference_patterns[ref_idx] != current_patterns[ref_idx]:
                    # Check if current pattern matches next reference pattern
                    if ref_idx + 1 < len(reference_patterns) and current_patterns[ref_idx] == reference_patterns[ref_idx + 1]:
                        missing_position = ref_idx
                        break
            
            # If we couldn't determine position, check for 'mostly_empty' column missing
            if missing_position is None:
                for ref_idx, ref_pattern in enumerate(reference_patterns):
                    if ref_pattern == 'mostly_empty':
                        # Check if this position seems to be missing
                        if ref_idx < len(current_patterns):
                            if current_patterns[ref_idx] != 'mostly_empty':
                                missing_position = ref_idx
                                break
                        else:
                            missing_position = ref_idx
                            break
            
            # Default to first mismatch or end
            if missing_position is None:
                missing_position = len(current_patterns)
            
            # Insert empty column at the missing position
            for row in data_rows:
                aligned_row = list(row)
                if len(aligned_row) < len(reference_patterns):
                    # Insert empty value at missing position
                    aligned_row.insert(missing_position, "")
                    # Pad remaining if needed
                    while len(aligned_row) < len(reference_patterns):
                        aligned_row.append("")
                aligned_rows.append(aligned_row)
            
            logging.warning(
                f"⚠️ Column alignment: Inserted empty column at position {missing_position} "
                f"(likely '{reference_header[missing_position] if missing_position < len(reference_header) else 'unknown'}'). "
                f"Column count adjusted from {len(current_patterns)} to {len(reference_patterns)}."
            )
            
            return aligned_rows

        for sec_idx, sec in enumerate(sections):
            # remove leading Table: lines and blanks
            sec_lines = [l for l in sec if not l.strip().startswith("Table:")]
            sec_lines = [l for l in sec_lines if l.strip()]
            if not sec_lines:
                continue

            # Consider only first N candidate lines for header detection
            N = min(6, len(sec_lines))
            header_idx = None

            for i in range(N):
                line = sec_lines[i].strip()
                toks = tokens_for_line(line)
                # require >2 tokens (at least 3 columns)
                if len(toks) <= 2:
                    continue

                # primary rule: none of tokens contain a digit
                if not any(re.search(r'\d', t) for t in toks):
                    header_idx = i
                    break

                # secondary: tokens mostly alphabetic (allow short numeric tokens like "NO")
                alpha_count = sum(1 for t in toks if re.match(r'^[A-Za-z\.\s]+$', t.strip()))
                if alpha_count >= max(2, len(toks) - 1):
                    header_idx = i
                    break

            # fallback uppercase-all-tokens heuristic (common in OCR headers)
            if header_idx is None:
                for i in range(N):
                    toks = tokens_for_line(sec_lines[i])
                    if len(toks) > 2 and all(any(c.isalpha() for c in t) and t.strip() == t.strip().upper() for t in toks if t.strip()):
                        header_idx = i
                        break

            # Process based on whether header was found
            if header_idx is None:
                # No header detected - continuation table
                if last_header is None:
                    # Can't process without a reference header
                    logging.warning(f"⚠️ Section {sec_idx}: No header detected and no reference header available. Skipping section.")
                    continue
                
                header = last_header
                data_lines = sec_lines
                
                # Parse data rows
                parsed_data_rows = []
                for l in data_lines:
                    s = l.strip()
                    if not s:
                        continue
                    low = s.lower()
                    if low.startswith("page total") or low.startswith("page") or low.startswith("total") or "page total" in low:
                        continue
                    parsed_data_rows.append(tokens_for_line(l))
                
                if not parsed_data_rows:
                    continue
                
                # Check column count
                current_col_count = len(parsed_data_rows[0]) if parsed_data_rows else 0
                
                if current_col_count < last_column_count:
                    # Fewer columns detected - need alignment
                    logging.warning(
                        f"⚠️ Section {sec_idx}: Continuation table with fewer columns detected. "
                        f"Expected {last_column_count}, found {current_col_count}."
                    )
                    
                    # Analyze current data patterns
                    current_patterns = analyze_column_patterns(parsed_data_rows)
                    
                    # Get reference patterns from last header
                    reference_header_tokens = tokens_for_line(last_header)
                    reference_patterns = get_reference_patterns(reference_header_tokens)
                    
                    logging.info(
                        f"ℹ️ Section {sec_idx} patterns: Current={current_patterns}, Reference={reference_patterns}"
                    )
                    
                    # Align columns
                    aligned_data_rows = align_columns(
                        parsed_data_rows, 
                        current_patterns, 
                        reference_header_tokens, 
                        reference_patterns
                    )
                    
                    # Create DataFrame with aligned data using csv.writer to preserve quotes
                    output = StringIO()
                    writer = csv.writer(output)
                    writer.writerow(reference_header_tokens)
                    writer.writerows(aligned_data_rows)
                    csv_chunk = output.getvalue()
                else:
                    # Same column count - use as is
                    csv_chunk = "\n".join([last_header] + data_lines)
                
            else:
                # Header detected
                header = sec_lines[header_idx].strip()
                last_header = header
                header_tokens = tokens_for_line(header)
                last_column_count = len(header_tokens)
                
                logging.info(f"✅ Section {sec_idx}: Header detected with {last_column_count} columns: {header_tokens}")
                
                data_lines = sec_lines[header_idx + 1 :]
                
                # filter out empty and summary/footer lines
                filtered = []
                for l in data_lines:
                    s = l.strip()
                    if not s:
                        continue
                    low = s.lower()
                    if low.startswith("page total") or low.startswith("page") or low.startswith("total") or "page total" in low:
                        continue
                    filtered.append(l)

                if not filtered:
                    continue

                csv_chunk = "\n".join([header] + filtered)

            # Parse CSV chunk with pandas
            try:
                df_chunk = pd.read_csv(StringIO(csv_chunk), engine="python")
            except Exception as e:
                logging.warning(f"⚠️ Section {sec_idx}: pandas parsing failed, using manual normalization. Error: {e}")
                reader = csv.reader(StringIO(csv_chunk))
                rows = list(reader)
                if not rows:
                    continue
                header_row = [h.strip() for h in rows[0]]
                ncols = len(header_row)
                normalized = []
                for r in rows[1:]:
                    if len(r) < ncols:
                        r = r + [""] * (ncols - len(r))
                    elif len(r) > ncols:
                        r = r[: ncols - 1] + [",".join(r[ncols - 1 :])]
                    normalized.append([c.strip() for c in r])
                df_chunk = pd.DataFrame(normalized, columns=header_row)

            dfs.append(df_chunk)

        if not dfs:
            raise Exception("No table data found in CSV text")

        df_all = pd.concat(dfs, ignore_index=True, sort=False).fillna("")
        df_all.columns = [str(c).strip() for c in df_all.columns]
        return df_all
    
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