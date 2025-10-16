"""
Login window with card-based UI design
"""

import os
import logging
from datetime import datetime, timedelta
import requests
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, 
    QPushButton, QHBoxLayout, QMessageBox, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont

from config import STYLE_SHEET, AUTH_SERVER_URL


class LoginWindow(QMainWindow):
    """Full login screen window with card-based design. Emits `authenticated` with username on success."""
    authenticated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bank Statement Processor - Login")
        self.setMinimumSize(900, 600)
        
        # Apply the same style sheet as main window
        self.setStyleSheet(STYLE_SHEET)
        
        # Security features
        self._failed_attempts = 0
        self._lockout_until = None
        
        self._setup_ui()

    def _setup_ui(self):
        """Setup the login user interface with centered card"""
        # Central widget with background
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout - this will hold the card in center
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create the login card
        self._create_login_card(main_layout)

    def _create_login_card(self, parent_layout):
        """Create the centered login card"""
        # Wrapper to center the card
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add vertical stretch before card
        wrapper_layout.addStretch()
        
        # Horizontal layout to center the card
        h_layout = QHBoxLayout()
        h_layout.addStretch()
        
        # The login card frame
        card = QFrame()
        card.setObjectName("loginCard")
        card.setFixedWidth(450)
        card.setStyleSheet("""
            QFrame#loginCard {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
                padding: 0px;
            }
        """)
        
        # Card layout
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(20)
        
        # App title/logo section
        title_label = QLabel("Bank Statement Processor")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        card_layout.addWidget(title_label)
        
        # Subtitle
        subtitle = QLabel("Sign in to continue")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; font-size: 14px; margin-bottom: 20px;")
        card_layout.addWidget(subtitle)
        
        # Username section
        username_label = QLabel("Username")
        username_label.setStyleSheet("color: #34495e; font-weight: 500; font-size: 13px;")
        card_layout.addWidget(username_label)
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter your username")
        self.user_input.setMinimumHeight(40)
        self.user_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                font-size: 14px;
                background-color: #f8f9fa;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: white;
            }
        """)
        card_layout.addWidget(self.user_input)
        
        # Password section
        password_label = QLabel("Password")
        password_label.setStyleSheet("color: #34495e; font-weight: 500; font-size: 13px; margin-top: 10px;")
        card_layout.addWidget(password_label)
        
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pass_input.setPlaceholderText("Enter your password")
        self.pass_input.setMinimumHeight(40)
        self.pass_input.returnPressed.connect(self._on_login)
        self.pass_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                font-size: 14px;
                background-color: #f8f9fa;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: white;
            }
        """)
        card_layout.addWidget(self.pass_input)
        
        # Status/Error label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            color: #e74c3c; 
            font-size: 12px;
            padding: 8px;
            background-color: #fadbd8;
            border-radius: 4px;
        """)
        self.status_label.hide()
        card_layout.addWidget(self.status_label)
        
        # Buttons section
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
            }
            QPushButton:pressed {
                background-color: #bdc3c7;
            }
        """)
        self.clear_btn.clicked.connect(self._on_clear)
        button_layout.addWidget(self.clear_btn)
        
        self.login_btn = QPushButton("Login")
        self.login_btn.setMinimumHeight(40)
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #ecf0f1;
            }
        """)
        self.login_btn.clicked.connect(self._on_login)
        button_layout.addWidget(self.login_btn)
        
        card_layout.addLayout(button_layout)
        
        # Add card to horizontal layout
        h_layout.addWidget(card)
        h_layout.addStretch()
        
        # Add horizontal layout to wrapper
        wrapper_layout.addLayout(h_layout)
        
        # Add vertical stretch after card
        wrapper_layout.addStretch()
        
        # Add wrapper to main layout
        parent_layout.addWidget(wrapper)

    def _on_login(self):
        """Handle login button click"""
        # Check for lockout
        if self._lockout_until:
            remaining = (self._lockout_until - datetime.now()).total_seconds()
            if remaining > 0:
                self._show_error(f"Too many attempts. Try again in {int(remaining)}s")
                return
            else:
                self._lockout_until = None
                self._failed_attempts = 0
                self._hide_error()

        username = self.user_input.text().strip()
        password = self.pass_input.text()
        
        if not username or not password:
            self._show_error("Please enter both username and password")
            return

        try:
            if self.validate_credentials(username, password):
                logging.info(f"Successful login for user: {username}")
                self._clear_sensitive_data()
                self.authenticated.emit(username)
                self.close()
            else:
                self._failed_attempts += 1
                logging.warning(f"Failed login attempt for user: {username}")
                
                # Implement progressive lockout
                if self._failed_attempts >= 5:
                    self._lockout_until = datetime.now() + timedelta(seconds=30)
                    self._show_error("Too many failed attempts. Locked for 30 seconds.")
                    self.login_btn.setEnabled(False)
                    QTimer.singleShot(30000, self._unlock)
                else:
                    remaining = 5 - self._failed_attempts
                    self._show_error(f"Invalid credentials. {remaining} attempt(s) remaining.")
        finally:
            # Always clear password field after attempt
            self.pass_input.clear()

    def _unlock(self):
        """Unlock login after timeout"""
        self.login_btn.setEnabled(True)
        self._hide_error()
        self._lockout_until = None
        self._failed_attempts = 0

    def _on_clear(self):
        """Handle clear button click"""
        self._clear_sensitive_data()
        # self.close()

    def _show_error(self, message: str):
        """Show error message"""
        self.status_label.setText(message)
        self.status_label.show()

    def _hide_error(self):
        """Hide error message"""
        self.status_label.hide()
        self.status_label.setText("")

    def _clear_sensitive_data(self):
        """Clear sensitive input fields"""
        self.user_input.clear()
        self.pass_input.clear()
        self._hide_error()

    def closeEvent(self, event):
        """Ensure sensitive data is cleared when window closes"""
        self._clear_sensitive_data()
        super().closeEvent(event)

    def validate_credentials(self, username: str, password: str) -> bool:
        """
        Validate credentials against authentication server.
        
        Sends credentials to the Flask authentication server over HTTPS.
        Returns True if authentication successful, False otherwise.
        """
        try:
            # Send credentials to server
            response = requests.post(
                f"{AUTH_SERVER_URL}/login",
                json={
                    'username': username,
                    'password': password
                },
                timeout=10,  # 10 second timeout
                verify=True,  # Verify SSL certificate
                headers={'Content-Type': 'application/json'}
            )
            
            # Log the response (never log password)
            if response.status_code == 200:
                logging.info(f"Authentication successful for user: {username}")
                return True
            elif response.status_code == 401:
                logging.warning(f"Authentication failed for user: {username}")
                return False
            else:
                logging.error(f"Unexpected response from auth server: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            logging.error("Authentication request timed out")
            QMessageBox.critical(
                self,
                "Connection Timeout",
                "Could not connect to authentication server.\nPlease check your internet connection."
            )
            return False
            
        except requests.exceptions.ConnectionError:
            logging.error("Could not connect to authentication server")
            QMessageBox.critical(
                self,
                "Connection Error",
                "Could not connect to authentication server.\nPlease check your internet connection."
            )
            return False
            
        except requests.exceptions.SSLError as e:
            logging.error(f"SSL certificate verification failed: {e}")
            QMessageBox.critical(
                self,
                "Security Error",
                "SSL certificate verification failed.\nThis could indicate a security risk."
            )
            return False
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Authentication request error: {e}")
            QMessageBox.critical(
                self,
                "Authentication Error",
                "An error occurred during authentication.\nPlease try again later."
            )
            return False
            
        except Exception as e:
            logging.exception(f"Unexpected error during authentication: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                "An unexpected error occurred"
            )
            return False