# file_handler.py
"""
File handling operations including upload, validation, and processing
"""

import os
import logging
from typing import Union, List, Tuple, Optional
from PyQt6.QtWidgets import QFileDialog, QWidget


class FileHandler:
    """Class for handling file operations"""
    
    # Supported file extensions
    PDF_EXTENSIONS = ("*.pdf",)
    EXCEL_EXTENSIONS = ("*.xlsx", "*.xls")
    IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff")
    
    @staticmethod
    def select_pdf_file(parent: QWidget) -> Optional[str]:
        """
        Open file dialog to select a PDF file
        
        Args:
            parent: Parent widget for the dialog
            
        Returns:
            Optional[str]: Selected file path or None if cancelled
        """
        file_path, _ = QFileDialog.getOpenFileName(
            parent, 
            "Select PDF File", 
            "", 
            f"PDF Files ({' '.join(FileHandler.PDF_EXTENSIONS)})"
        )
        
        if file_path:
            logging.info("PDF selected: %s", file_path)
            return file_path
        
        return None
    
    @staticmethod
    def select_excel_file(parent: QWidget) -> Optional[str]:
        """
        Open file dialog to select an Excel file
        
        Args:
            parent: Parent widget for the dialog
            
        Returns:
            Optional[str]: Selected file path or None if cancelled
        """
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Select Excel File",
            "",
            f"Excel Files ({' '.join(FileHandler.EXCEL_EXTENSIONS)})"
        )
        
        if file_path:
            logging.info("Excel selected: %s", file_path)
            return file_path
        
        return None
    
    @staticmethod
    def select_image_files(parent: QWidget) -> List[str]:
        """
        Open file dialog to select image files (multiple selection)
        
        Args:
            parent: Parent widget for the dialog
            
        Returns:
            List[str]: List of selected file paths (empty if cancelled)
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            parent,
            "Select Image Files",
            "",
            f"Image Files ({' '.join(FileHandler.IMAGE_EXTENSIONS)})"
        )
        
        if file_paths:
            logging.info("Images selected: %s", file_paths)
            return file_paths
        
        return []
    
    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """
        Check if file exists
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if file exists, False otherwise
        """
        exists = os.path.isfile(file_path)
        if not exists:
            logging.warning("File does not exist: %s", file_path)
        return exists
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Get file extension from path
        
        Args:
            file_path: Path to file
            
        Returns:
            str: File extension (lowercase, without dot)
        """
        return os.path.splitext(file_path)[1].lower().lstrip('.')
    
    @staticmethod
    def is_excel_file(file_path: str) -> bool:
        """
        Check if file is an Excel file based on extension
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if Excel file, False otherwise
        """
        extension = FileHandler.get_file_extension(file_path)
        return extension in ['xlsx', 'xls']
    
    @staticmethod
    def is_pdf_file(file_path: str) -> bool:
        """
        Check if file is a PDF file based on extension
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if PDF file, False otherwise
        """
        extension = FileHandler.get_file_extension(file_path)
        return extension == 'pdf'
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """
        Check if file is an image file based on extension
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if image file, False otherwise
        """
        extension = FileHandler.get_file_extension(file_path)
        return extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']
    
    @staticmethod
    def get_file_display_name(file_path: Union[str, List[str]]) -> str:
        """
        Get display-friendly name for files
        
        Args:
            file_path: Single file path or list of file paths
            
        Returns:
            str: Display name for UI
        """
        if isinstance(file_path, list):
            if len(file_path) == 1:
                return f"Uploaded file: {os.path.basename(file_path[0])}"
            else:
                return f"Uploaded {len(file_path)} files"
        else:
            return f"Uploaded file: {os.path.basename(file_path)}"
    
    @staticmethod
    def get_file_type_display(file_path: Union[str, List[str]]) -> str:
        """
        Get file type for display purposes
        
        Args:
            file_path: Single file path or list of file paths
            
        Returns:
            str: File type description
        """
        if isinstance(file_path, list):
            if len(file_path) == 1:
                if FileHandler.is_image_file(file_path[0]):
                    return "image"
                else:
                    return "file"
            else:
                return f"{len(file_path)} image(s)"
        else:
            if FileHandler.is_excel_file(file_path):
                return "Excel"
            elif FileHandler.is_pdf_file(file_path):
                return "PDF"
            elif FileHandler.is_image_file(file_path):
                return "image"
            else:
                return "file"