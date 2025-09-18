# ui_components.py
"""
Custom UI components and widgets
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen


class LoaderWidget(QWidget):
    """Custom loading spinner widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.running = False
        
        # Increase widget size to accommodate spinner and label
        self.setFixedSize(120, 120)
        
        # Create layout for better organization
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Create spinner container
        self.spinner_container = QWidget()
        self.spinner_container.setFixedSize(80, 80)
        layout.addWidget(self.spinner_container, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Add label with proper width
        self.label = QLabel("Processing file(s)...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #0d6efd; font-weight: bold;")
        layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.hide()

    def paintEvent(self, event):
        """Paint the spinning loader"""
        if not self.running:
            return

        # Paint directly on the spinner container widget
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(QColor("#0d6efd"))
        pen.setWidth(5)
        painter.setPen(pen)
        
        # Calculate the rect relative to the spinner container's position
        container_pos = self.spinner_container.pos()
        rect = self.spinner_container.rect()
        rect.moveTopLeft(container_pos)
        rect.adjust(10, 10, -10, -10)
        
        # Draw arc in the adjusted rect
        painter.drawArc(rect, self.angle * 16, 90 * 16)

    def start(self):
        """Start the loading animation"""
        self.running = True
        self.show()
        self.timer.start(50)

    def rotate(self):
        """Rotate the spinner"""
        self.angle = (self.angle - 10) % 360
        self.update()

    def stop(self):
        """Stop the loading animation"""
        self.running = False
        self.timer.stop()
        self.hide()


class WorkerThread(QThread):
    """Generic worker thread for background processing"""
    
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Execute the worker function"""
        try:
            result = self.function(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))