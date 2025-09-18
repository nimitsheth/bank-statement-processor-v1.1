# main.py
"""
Main entry point for the Bank Statement Processor application
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication
from main_window import BankStatementProcessor


def setup_logging():
    """Setup application logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def main():
    """Main application entry point"""
    # Setup logging
    setup_logging()
    logging.info("Starting Bank Statement Processor App")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = BankStatementProcessor()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()