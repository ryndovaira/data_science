
# IntelliJ Plugin for Developer Insights

This project aims to create an IntelliJ-based plugin (compatible with PyCharm, WebStorm, IntelliJ IDEA) to assist developers by providing insights, file reviews, and extensive checks for their projects. Powered by the ChatGPT API and **LangChain**, this plugin delivers enhanced code quality and productivity features.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
   - [Installation](#installation)
4. [Usage](#usage)
   - [Plugin Workflow](#plugin-workflow)
   - [Uploading Directories](#uploading-directories)
5. [Features](#features)
6. [Best Practices for File Uploads](#best-practices-for-file-uploads)
7. [Future Work](#future-work)
8. [Contributing](#contributing)
9. [About the Author](#about-the-author)

---

## Overview

The IntelliJ Plugin for Developer Insights enables developers to:
- Select project files or entire trees for review.
- Request targeted checks such as:
  - Code review.
  - Code quality and style analysis.
  - Auto-generated `README.md` files.
- Leverage the ChatGPT API for real-time feedback and insights.

This plugin aims to integrate seamlessly into the IntelliJ environment, offering a natural extension of developer workflows with AI-driven insights.

---

## Project Structure

All development for the plugin takes place in the `intellij_ai_assistant/` directory:

```
deep_learning/
├── intellij_ai_assistant/
│   ├── README.md             # Plugin-specific documentation
│   ├── __init__.py           # Initialization file for the plugin
│   ├── config/               # Configuration files
│   ├── src/                  # Core plugin source code (Kotlin/Python)
│   ├── resources/            # Plugin resources (e.g., icons, templates)
│   └── tests/                # Unit tests for plugin functionality
```

Other directories under `deep_learning/` are unrelated to this project.

---

## Requirements

Ensure you have the following dependencies and tools installed:

- **Programming Language**: Kotlin (or Python if feasible)
- **Framework**: LangChain
- **ChatGPT API**: Enabled for large-context responses
- **ASGI Server**: Uvicorn for running FastAPI applications
- IntelliJ IDEA or compatible JetBrains IDEs

### Installation

To set up the project:
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo>/intellij-plugin.git
   ```

2. Install backend dependencies:
   ```bash
   pip install fastapi uvicorn langchain
   ```

3. Open the project in IntelliJ IDEA.

---

## Usage

### Plugin Workflow

1. **Activate the Plugin**:
   - Once installed, activate the plugin from the IntelliJ Plugin Manager.

2. **Select Project Root**:
   - Navigate to the desired project root and initiate the plugin.

3. **Choose Files**:
   - Use the plugin interface to select specific files or an entire project tree.

4. **Select Action**:
   - Options include:
     - Code Review.
     - Code Quality and Style Analysis.
     - Auto-generation of `README.md` files.

5. **Receive Insights**:
   - The plugin communicates with a server that uses ChatGPT API to process selected files and deliver insights.

### Uploading Directories

For projects with multiple files or directories, compress them into a `.zip` or `.tar` file before uploading.

#### Upload Example (FastAPI Endpoint)
1. Compress the directory on the client side:
   ```bash
   zip -r project.zip /path/to/project
   ```

2. Send the compressed file to the API:
   ```bash
   curl -X POST "http://127.0.0.1:8000/upload-zip/" -F "file=@project.zip"
   ```

3. The backend extracts and processes the contents.

---

## Features

- **Project Tree Analysis**:
  - Scans the project and generates a hierarchical file structure.

- **Flexible File Selection**:
  - Allows developers to choose specific files or directories.

- **Customizable Checks**:
  - Includes options for code reviews, quality checks, and documentation.

- **ChatGPT API Integration**:
  - Provides large-context responses to deliver high-quality insights.

---

## Best Practices for File Uploads

1. **Use Compression for Directories**:
   - Upload directories as `.zip` or `.tar` archives to preserve structure and reduce network overhead.

2. **Set Reasonable Size Limits**:
   - Limit upload sizes to avoid overwhelming the server (e.g., 50 MB).

3. **Validate Archive Contents**:
   - Check for allowed file types and structures before processing.

4. **Handle Errors Gracefully**:
   - Notify users of issues such as corrupted files or unsupported formats.

5. **Asynchronous Processing**:
   - Use asynchronous extraction and processing to ensure responsiveness.

---

## Future Work

1. **Expand Supported Checks**:
   - Add static code analysis and security checks.

2. **Integrate Additional Models**:
   - Support other language models like **LLaMA** or **Anthropic’s Claude**.

3. **UI Enhancements**:
   - Add a more interactive and user-friendly plugin interface.

4. **Explainable AI**:
   - Offer explainability for the plugin’s insights to help developers understand feedback.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## About the Author

This project was created by **Irina Ryndova**, a Senior Data Scientist passionate about creating tools to enhance developer productivity.

- GitHub: [ryndovaira](https://github.com/ryndovaira)
- Email: [ryndovaira@gmail.com](mailto:ryndovaira@gmail.com)

Feel free to reach out for collaborations or feedback.

---
