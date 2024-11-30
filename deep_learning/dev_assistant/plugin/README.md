
# IntelliJ Plugin for Dev Assistant

This directory contains the IntelliJ-based plugin for the **Dev Assistant** project. 
The plugin integrates IntelliJ-based IDEs (such as PyCharm, WebStorm, and IntelliJ IDEA) 
with the backend server to provide AI-powered insights into selected project files.

---

## Overview

The plugin is designed to:
- Allow developers to **select files or directories** in their projects for analysis.
- Automatically **compress selected files** into a `.zip` archive for efficient transfer.
- Send the archive to the backend **FastAPI server** for processing.
- Display the analysis results (e.g., code reviews, quality checks) directly in the IDE.

This plugin bridges the IDE and the backend, enabling seamless interaction and enhanced developer productivity.

---

## Project Structure

```
plugin/
├── __init__.py           # Placeholder for Python compatibility
├── README.md             # Plugin documentation
└── (future files and directories for plugin code)
```

---

## Features

1. **File and Directory Selection**:
   - Use the plugin UI to select specific files or directories within the IDE.

2. **File Compression**:
   - Automatically compress selected files into `.zip` archives.

3. **Backend Communication**:
   - Send `.zip` files to the backend via HTTP POST requests.
   - Handle backend responses and display insights in the IDE.

4. **Integration with IntelliJ IDEs**:
   - Compatible with IntelliJ IDEA, PyCharm, and WebStorm.

---

## Development Workflow

### Initial Implementation
1. Create a basic UI in the plugin to allow file/directory selection.
2. Implement functionality to compress selected files into a `.zip` archive.

### Backend Integration
1. Add support for sending `.zip` files to the backend FastAPI server.
2. Handle responses from the backend and display results in the IDE.

### Enhancements
1. Add real-time feedback for large file uploads.
2. Include additional UI features like progress bars and file previews.

---

## Requirements

- **Kotlin**: The plugin is developed in Kotlin to leverage IntelliJ's SDK.
- **IntelliJ Platform SDK**: For plugin development.
- **Backend**: The plugin relies on a running FastAPI server for analysis.

---

## Testing

- Test the plugin on **IntelliJ IDEA Community Edition** and **PyCharm Professional Edition**.
- Verify file compression, API communication, and result rendering in the IDE.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## Future Plans

- Support analysis of individual code snippets without file uploads.
- Add AI-generated suggestions for improving code quality and performance.
- Include advanced error handling and feedback mechanisms.

---

## Contact

For any questions or suggestions, please reach out via the main project's contact information.
