import os


def save_file(file):
    """
    Saves an uploaded file to the temporary directory.
    """
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


def read_extracted_files(directory):
    """
    Reads all files from an extracted directory and concatenates their content.
    """
    content = ""
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content += f.read() + "\n"
    return content


def truncate_content(content: str, token_limit: int = 1000) -> str:
    """
    Truncates content to a specified token limit.

    Args:
        content (str): Full text content to truncate.
        token_limit (int): Maximum number of tokens (characters) allowed.

    Returns:
        str: Truncated content.
    """
    return content[:token_limit]
