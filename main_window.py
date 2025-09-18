# main_window.py
"""
Main application window and UI logic
"""

import os
import logging
from typing import Union, List, Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QLineEdit, QMessageBox, QTableWidget, 
    QTableWidgetItem, QFrame
)
from PyQt6.QtCore import Qt

from config import STYLE_SHEET, WINDOW_TITLE, MIN_WINDOW_SIZE, GEMINI_PROMPT
from ui_components import LoaderWidget, WorkerThread
from file_handler import FileHandler
from data_processor import DataProcessor
from gemini_service import GeminiService


class BankStatementProcessor(QMainWindow):
    """Main application window for processing bank statements"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(*MIN_WINDOW_SIZE)
        self.setStyleSheet(STYLE_SHEET)
        
        # Initialize services
        self.gemini_service = GeminiService()
        self.file_handler = FileHandler()
        self.data_processor = DataProcessor()
        
        # Initialize state variables
        self.worker = None
        self.processing = False
        self.file_path = None
        self.file_paths = None
        self.df = None
        
        # Setup UI
        self._setup_ui()
        self._connect_signals()
        
        logging.info("Bank Statement Processor initialized")

    def _setup_ui(self):
        """Setup the main user interface"""
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Create UI components
        self._create_top_frame(main_layout)
        self._create_file_info_label(main_layout)
        self._create_table_widget(main_layout)
        self._create_bottom_frame(main_layout)
        
        # Create loader widget
        self.loader = LoaderWidget(self.table)
        self.loader.hide()

    def _create_top_frame(self, parent_layout):
        """Create the top frame with filename input and process buttons"""
        top_frame = QFrame()
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Filename input components
        filename_label = QLabel("Excel File Name:")
        self.filename_entry = QLineEdit()
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.convert_button = QPushButton("Convert to Excel")
        self.convert_button.setEnabled(False)

        # Add components to layout
        top_layout.addWidget(filename_label)
        top_layout.addWidget(self.filename_entry)
        top_layout.addWidget(self.process_button)
        top_layout.addWidget(self.convert_button)
        top_layout.addStretch()

        parent_layout.addWidget(top_frame)

    def _create_file_info_label(self, parent_layout):
        """Create the file information label"""
        self.file_label = QLabel("No file uploaded")
        self.file_label.setStyleSheet("color: gray; font-style: italic;")
        parent_layout.addWidget(self.file_label)

    def _create_table_widget(self, parent_layout):
        """Create the data table widget"""
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        parent_layout.addWidget(self.table)

    def _create_bottom_frame(self, parent_layout):
        """Create the bottom frame with upload buttons"""
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(25)
        
        # Upload buttons
        self.pdf_button = QPushButton("Upload PDF")
        self.pdf_button.setObjectName("pdf_button")
        self.excel_button = QPushButton("Upload Excel")
        self.excel_button.setObjectName("excel_button")
        self.image_button = QPushButton("Upload Image")
        self.image_button.setObjectName("image_button")
        self.clear_button = QPushButton("Clear Data")
        self.clear_button.setObjectName("clear_button")

        # Configure buttons
        for btn in [self.pdf_button, self.excel_button, self.image_button, self.clear_button]:
            btn.setMinimumWidth(120)
            bottom_layout.addWidget(btn)

        bottom_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        parent_layout.addWidget(bottom_frame)

    def _connect_signals(self):
        """Connect UI signals to handlers"""
        self.process_button.clicked.connect(self.process_file)
        self.convert_button.clicked.connect(self.save_to_excel)
        self.pdf_button.clicked.connect(self.upload_pdf)
        self.excel_button.clicked.connect(self.upload_excel)
        self.image_button.clicked.connect(self.upload_image)
        self.clear_button.clicked.connect(self._reset_application_state)

    def center_loader(self):
        """Center the loader widget on the table"""
        if self.loader and self.table:
            # Use the viewport for correct centering inside the table
            viewport = self.table.viewport()
            vp_width = viewport.width()
            vp_height = viewport.height()
            loader_size = self.loader.size()
            x = (vp_width - loader_size.width()) // 2 + 100  # Move 100px right
            y = (vp_height - loader_size.height()) // 2 - 20  # Move 20px up
            self.loader.move(x, y)

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        self.center_loader()

    def closeEvent(self, event):
        """Handle application closure"""
        if self.processing:
            reply = QMessageBox.question(
                self,
                "Processing in Progress",
                "A file is currently being processed. Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.worker and self.worker.isRunning():
                    self.worker.quit()
                    self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # File Upload Methods
    def upload_pdf(self):
        """Handle PDF file upload"""
        file_path = self.file_handler.select_pdf_file(self)
        if file_path:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.convert_button.setEnabled(False)
            self._clear_previous_files()
            self.file_path = file_path
            self._update_file_label(file_path, "PDF")
            self._enable_processing()
            logging.info("PDF uploaded: %s", file_path)

    def upload_excel(self):
        """Handle Excel file upload"""
        file_path = self.file_handler.select_excel_file(self)
        if file_path:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.convert_button.setEnabled(False)
            self._clear_previous_files()
            self.file_path = file_path
            self._update_file_label(file_path, "Excel")
            self._enable_processing()
            logging.info("Excel uploaded: %s", file_path)

    def upload_image(self):
        """Handle image file(s) upload"""
        file_paths = self.file_handler.select_image_files(self)
        if file_paths:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.convert_button.setEnabled(False)
            self._clear_previous_files()
            self.file_paths = file_paths
            
            # Update label based on number of files
            if len(file_paths) == 1:
                label_text = f"Uploaded image: {os.path.basename(file_paths[0])}"
            else:
                label_text = f"Uploaded {len(file_paths)} image(s)"
                
            self.file_label.setText(label_text)
            self.file_label.setStyleSheet("color: black;")
            self._enable_processing()
            logging.info("Images uploaded: %s", file_paths)

    def _clear_previous_files(self):
        """Clear previously uploaded files"""
        self.file_path = None
        self.file_paths = None

    def _update_file_label(self, file_path: str, file_type: str):
        """Update the file information label"""
        filename = os.path.basename(file_path)
        self.file_label.setText(f"Uploaded {file_type}: {filename}")
        self.file_label.setStyleSheet("color: black;")

    def _enable_processing(self):
        """Enable the process button"""
        self.process_button.setEnabled(True)

    # File Processing Methods
    def process_file(self):
        """Process the uploaded file(s)"""
        if self.processing:
            QMessageBox.warning(
                self, 
                "Processing in Progress", 
                "Please wait for the current process to complete."
            )
            return

        # Determine file input and type
        file_input, input_type = self._get_file_input()
        if not file_input:
            QMessageBox.warning(
                self, 
                "No File", 
                "Please upload a PDF, Excel, or image(s) first."
            )
            logging.warning("Process attempted without file(s)")
            return

        # Confirm processing with user
        if not self._confirm_processing(file_input, input_type):
            logging.info("Processing cancelled by user")
            return

        # Start processing
        self._start_processing(file_input, input_type)

    def _get_file_input(self):
        """Get the current file input and determine its type"""
        if hasattr(self, "file_paths") and self.file_paths:
            return self.file_paths, "images"
        elif hasattr(self, "file_path") and self.file_path:
            if self.file_handler.is_excel_file(self.file_path):
                return self.file_path, "excel"
            else:
                return self.file_path, "pdf"
        return None, None

    def _confirm_processing(self, file_input, input_type):
        """Ask user to confirm processing"""
        if input_type == "images":
            msg = f"Are you sure you want to process {len(file_input)} image(s)?"
        else:
            filename = os.path.basename(file_input)
            msg = f"Are you sure you want to process the file {filename}?"
        
        reply = QMessageBox.question(
            self, 
            "Processing", 
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        return reply == QMessageBox.StandardButton.Yes

    def _start_processing(self, file_input, input_type):
        """Start the file processing in background thread"""
        try:
            self.process_button.setEnabled(False)
            self.processing = True
            self.loader.start()

            # Clean up any existing worker thread
            if self.worker and self.worker.isRunning():
                self.worker.quit()
                self.worker.wait()

            # Create processing function based on input type
            if input_type == "excel":
                process_func = self.data_processor.process_excel_file
            else:
                process_func = self._process_with_gemini

            # Create and start worker thread
            self.worker = WorkerThread(process_func, file_input)
            self.worker.finished.connect(self.handle_processing_complete)
            self.worker.error.connect(self.handle_processing_error)
            self.worker.start()

        except Exception as e:
            self._stop_processing()
            logging.exception("Error starting processing thread")
            QMessageBox.critical(
                self, 
                "Error", 
                f"Error starting processing:\n{str(e)}"
            )

    def _process_with_gemini(self, file_input):
        """Process files using Gemini AI service"""
        try:
            # Get response from Gemini
            csv_text = self.gemini_service.send_to_gemini(file_input, GEMINI_PROMPT)
            
            # Parse CSV text into DataFrame
            df = self.data_processor.csv_to_dataframe(csv_text)
            return df
            
        except Exception as e:
            logging.error("Error processing with Gemini: %s", str(e))
            raise Exception(f"Error processing with Gemini: {str(e)}")

    def handle_processing_complete(self, df):
        """Handle successful processing completion"""
        self.df = df
        self.show_dataframe(df)
        self._stop_processing()
        self.process_button.setEnabled(False)
        self.convert_button.setEnabled(True)
        self._cleanup_worker()
        
    def handle_processing_error(self, error_message):
        """Handle processing errors"""
        self._stop_processing()
        self._cleanup_worker()
        QMessageBox.critical(self, "Error", str(error_message))

    def _stop_processing(self):
        """Stop processing and reset UI state"""
        self.loader.stop()
        self.process_button.setEnabled(True)
        self.processing = False

    def _cleanup_worker(self):
        """Clean up the worker thread"""
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None

    # Data Display Methods
    def show_dataframe(self, df):
        """Display DataFrame in the table widget"""
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)

        for row in range(len(df)):
            for col in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[row, col]))
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()

    # Excel Export Methods
    def save_to_excel(self):
        """Save processed data to Excel file"""
        if not hasattr(self, "df") or self.df is None:
            QMessageBox.warning(self, "No Data", "No data available to save.")
            return

        # Get output filename
        output_name = self.filename_entry.text().strip()
        if not output_name:
            QMessageBox.warning(self, "Warning", "Please enter a filename first.")
            return
        
        if not output_name.endswith(".xlsx"):
            output_name += ".xlsx"

        try:
            # Save to Excel
            success = self.data_processor.save_to_excel(self.df, output_name)
            
            if success:
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Excel saved as {output_name}"
                )
                self._reset_application_state()
                logging.info("Successfully saved Excel and reset application state")
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Error saving Excel file:\n{str(e)}"
            )
            logging.error("Error saving Excel file: %s", str(e))

    def _reset_application_state(self):
        """Reset application to initial state after successful save"""
        # Clear table
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        
        # Clear form fields
        self.filename_entry.clear()
        self.file_label.setText("No file uploaded")
        self.file_label.setStyleSheet("color: gray; font-style: italic;")
        
        # Reset buttons
        self.process_button.setEnabled(False)
        self.convert_button.setEnabled(False)
        
        # Clear data
        self.df = None
        self.file_path = None
        self.file_paths = None