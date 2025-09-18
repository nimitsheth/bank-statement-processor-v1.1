# Bank Statement Processor

A PyQt6-based application for processing bank statements using Google's Gemini AI service.

## Project Structure

The application has been refactored into multiple modules for better maintainability:

```
bank_statement_processor/
├── main.py                 # Application entry point
├── main_window.py          # Main application window and UI logic
├── config.py               # Configuration, constants, and styles
├── gemini_service.py       # Gemini AI service integration
├── data_processor.py       # Data processing and CSV handling
├── file_handler.py         # File operations and validation
├── ui_components.py        # Custom UI components (LoaderWidget, WorkerThread)
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Module Descriptions

### `main.py`
- Application entry point
- Sets up logging and launches the main window
- Handles application lifecycle

### `main_window.py`
- Contains the main `BankStatementProcessor` class
- Handles all UI interactions and user workflows
- Coordinates between different services
- Manages application state

### `config.py`
- Centralized configuration management
- Contains all constants, API keys, and styling
- Gemini prompt templates
- Column mapping dictionaries

### `gemini_service.py`
- Encapsulates all Gemini AI interactions
- Handles file uploads to Gemini
- Manages API configuration and authentication
- Provides clean interface for AI processing

### `data_processor.py`
- Handles all data processing operations
- CSV parsing and DataFrame operations
- Excel file processing and standardization
- Data format mapping and validation

### `file_handler.py`
- File selection and validation utilities
- Supports PDF, Excel, and image file operations
- File type detection and validation
- User-friendly file operation dialogs

### `ui_components.py`
- Custom UI widgets and components
- Loading spinner widget
- Background worker thread class
- Reusable UI elements

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python main.py
```

## Features

- **Multi-format Support**: Process PDF, Excel, and image files
- **AI Processing**: Uses Google Gemini AI for intelligent data extraction
- **Data Standardization**: Converts various bank statement formats to standardized CSV
- **Excel Export**: Save processed data to Excel files
- **User-friendly Interface**: Clean, modern PyQt6 interface
- **Background Processing**: Non-blocking file processing with progress indication

## Configuration

Set your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or modify the key directly in `config.py` (not recommended for production).

## Benefits of Refactored Structure

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Maintainability**: Easier to locate and modify specific functionality
3. **Testability**: Individual components can be tested in isolation
4. **Reusability**: Components can be reused across different parts of the application
5. **Readability**: Smaller, focused files are easier to understand
6. **Extensibility**: New features can be added without affecting existing code

## Adding New Features

- **New file formats**: Extend `FileHandler` class
- **New AI services**: Create new service classes similar to `GeminiService`
- **New data formats**: Extend `DataProcessor` with new mapping functions
- **New UI components**: Add to `ui_components.py`
- **Configuration changes**: Update `config.py`

This modular structure makes the application much easier to maintain, test, and extend while preserving all existing functionality.