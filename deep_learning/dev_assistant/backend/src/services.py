from pathlib import Path
import zipfile
from logging_config import setup_logger
from config import (
    OPENAI_API_KEY,
    OPENAI_PROJECT_ID,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    USE_REAL_OPENAI_API,
    DUMMY_RESPONSE,
)
from pydantic_models import ErrorResponse
from openai import OpenAI

logger = setup_logger(__name__)

# Define assistance options
ASSISTANCE_OPTIONS = {
    "code_review": "Provide a detailed code review for this file, highlighting best practices, potential issues, and suggestions for improvement.",
    "generate_readme": "Based on the provided code, generate a `README.md` that includes an overview, installation instructions, and examples of usage.",
    "explain_code": "Explain the functionality and purpose of this code in a simple, clear manner.",
    "check_docs": "Check this code for documentation completeness. Highlight functions/classes without adequate docstrings or comments.",
    "static_analysis": "Perform static code analysis on this file and list potential issues such as PEP8 violations, unused imports, and insecure code practices.",
    "suggest_improvements": "Suggest improvements for this code to enhance readability, maintainability, and efficiency.",
}

# Supported file types for validation
SUPPORTED_FILE_TYPES = [".zip"]


# Dependency injection for OpenAI client
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)


# Decorator for centralized error handling
def handle_errors(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as ve:
            logger.error(f"Validation Error: {ve}")
            return ErrorResponse(error=str(ve), success=False).dict()
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            return ErrorResponse(
                error="An unexpected error occurred.", details=str(e), success=False
            ).dict()

    return wrapper


# Function to validate file and assistance type
async def validate_file(file, assistance_type):
    if assistance_type not in ASSISTANCE_OPTIONS:
        raise ValueError(f"Invalid assistance type: {assistance_type}")

    if not any(file.filename.endswith(ext) for ext in SUPPORTED_FILE_TYPES):
        raise ValueError(
            f"Unsupported file type. Only the following are supported: {SUPPORTED_FILE_TYPES}"
        )

    file_content = await file.read()
    if not file_content:
        raise ValueError("Uploaded file is empty.")

    return file_content


# Function to extract ZIP contents
async def extract_zip(file):
    try:
        with zipfile.ZipFile(file.file, "r") as zip_ref:
            file_contents = {
                file_name: zip_ref.read(file_name).decode("utf-8")
                for file_name in zip_ref.namelist()
            }
            project_structure = "\n".join([f"- {name}" for name in file_contents.keys()])
            combined_files = "\n\n".join(
                [f"### {name}\n\n{content}" for name, content in file_contents.items()]
            )
        return project_structure, combined_files
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file uploaded. Please provide a valid ZIP file.")
    except Exception as e:
        raise ValueError(f"Error extracting ZIP file: {e}")


# Function to prepare messages for OpenAI API
async def prepare_messages(assistance_type, project_structure, combined_files):
    return [
        {"role": "system", "content": "You are a helpful assistant for developers."},
        {
            "role": "user",
            "content": (
                f"Assistance Type: {assistance_type}\n\n"
                f"Project Structure:\n{project_structure}\n\n"
                f"Files:\n{combined_files}\n\n"
                f"{ASSISTANCE_OPTIONS.get(assistance_type, 'No specific assistance requested.')}"
            ),
        },
    ]


# Function to call OpenAI API or return dummy data
async def call_openai_api(client, messages):
    if USE_REAL_OPENAI_API:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            raise ValueError("An error occurred while communicating with the OpenAI API.")
    else:
        logger.info("Using mock data (testing mode) instead of calling the OpenAI API.")
        return DUMMY_RESPONSE


# Main function to process and analyze the file
@handle_errors
async def process_and_analyze_file(file, assistance_type=None, save_to_disk=False, client=None):
    logger.info(f"Starting analysis for file: {file.filename}")

    # Use dependency injection for the OpenAI client
    client = client or get_openai_client()

    # Validate the file and assistance type
    file_content = await validate_file(file, assistance_type)

    # Optionally save file to disk
    if save_to_disk:
        temp_dir = Path(__file__).resolve().parents[2] / "debug"
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / file.filename
        try:
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            logger.info(f"File saved to {temp_file_path}")
        except Exception as e:
            logger.error(f"Error saving file to disk: {e}")
            raise

    # Extract project structure and file contents from ZIP
    project_structure, combined_files = await extract_zip(file)

    # Prepare messages for the OpenAI API
    messages = await prepare_messages(assistance_type, project_structure, combined_files)

    # Call the OpenAI API
    result = await call_openai_api(client, messages)

    logger.info(f"OpenAI response: {result}")

    return {
        "filename": file.filename,
        "analysis": {"summary": result},
        "success": True,
    }
