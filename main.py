# main.py
"""
Main entry point for the Bank Statement Processor application
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication
from main_window import BankStatementProcessor
from login_window import LoginWindow


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
    
    login = LoginWindow()
    main_window = None

    # When authenticated, create and show main window
    def on_auth(username):
        nonlocal main_window
        logging.info("User logged in: %s", username)
        main_window = BankStatementProcessor()
        main_window.show()

    login.authenticated.connect(on_auth)
    login.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()