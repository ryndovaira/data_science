from logging_config import setup_logger
from pathlib import Path

logger = setup_logger(__name__)


async def process_and_analyze_file(file, save_to_disk=False):
    """
    Processes the uploaded file and performs analysis.

    Args:
        file (UploadFile): The uploaded file to process.
        save_to_disk (bool): Whether to save the file to disk before processing.

    Returns:
        dict: A dictionary with the analysis result or an error message.
    """
    try:
        logger.info(f"Starting analysis for file: {file.filename}")

        # Read the file content into memory
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes from the file in memory.")

        # Optional: Save the file to disk if save_to_disk is True
        if save_to_disk:
            temp_dir = Path(__file__).resolve().parents[2] / "debug"
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / file.filename

            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)

            logger.info(f"File saved to {temp_file_path} for further processing.")

        # Simulated analysis logic (replace with actual implementation)
        if file.filename == "debug_files.zip":
            result = {"analysis": {"result": "Debug file analyzed successfully"}}
        else:
            result = {"analysis": {"result": "File analyzed successfully"}}

        logger.info(f"Analysis complete for file: {file.filename}")
        return {
            "filename": file.filename,
            "analysis": result["analysis"],
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        return {
            "error": "An error occurred during processing.",
            "details": str(e),
            "success": False,
        }
