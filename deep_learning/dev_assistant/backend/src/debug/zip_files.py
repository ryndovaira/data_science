import zipfile
from pathlib import Path


def zip_files(output_path, files_to_zip):
    """
    Create a ZIP archive from the specified list of files.

    Args:
        output_path (str or Path): The path where the ZIP file will be created.
        files_to_zip (list of str or Path): A list of file paths to include in the ZIP.

    Raises:
        ValueError: If the `files_to_zip` list is empty.
        Exception: For any other unexpected error during zipping.

    Notes:
        - This function resolves all file paths to absolute paths.
        - Missing files are skipped, and a message is printed for each missing file.
        - The ZIP file will be created even if some files are missing, as long as at least one valid file is found.
    """
    try:
        if not files_to_zip:
            raise ValueError("No files to zip.")

        with zipfile.ZipFile(output_path, "w") as zipf:
            for file in files_to_zip:
                abs_path = Path(file).resolve()
                if abs_path.exists():
                    zipf.write(abs_path, abs_path.name)
                    print(f"Added: {abs_path}")
                else:
                    print(f"File not found: {abs_path}")
        print(f"ZIP created: {output_path}")
    except Exception as e:
        print(f"Error creating ZIP: {e}")


if __name__ == "__main__":
    hands_on_keras_path = Path(__file__).resolve().parents[4] / "rnn" / "hands_on_keras"
    files = [
        hands_on_keras_path / "config.py",
        # hands_on_keras_path / "data_preprocessing.py",
        # hands_on_keras_path / "logger.py",
        hands_on_keras_path / "main.py",
        # hands_on_keras_path / "plotter.py",
        # hands_on_keras_path / "requirements.txt",
        # hands_on_keras_path / "saver.py",
        # hands_on_keras_path / "tuner.py",
        # hands_on_keras_path / "utils.py",
    ]

    # Define the output ZIP file path
    zip_output = Path(__file__).resolve().parent / "debug_files.zip"

    # Create the ZIP file
    zip_files(zip_output, files)
