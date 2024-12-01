from openai import OpenAI
from logging_config import setup_logger
from pathlib import Path
import zipfile
from config import OPENAI_API_KEY, OPENAI_PROJECT_ID

logger = setup_logger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)

# Define assistance options
ASSISTANCE_OPTIONS = {
    "code_review": "Provide a detailed code review for this file, highlighting best practices, potential issues, and suggestions for improvement.",
    "generate_readme": "Based on the provided code, generate a `README.md` that includes an overview, installation instructions, and examples of usage.",
    "explain_code": "Explain the functionality and purpose of this code in a simple, clear manner.",
    "check_docs": "Check this code for documentation completeness. Highlight functions/classes without adequate docstrings or comments.",
    "static_analysis": "Perform static code analysis on this file and list potential issues such as PEP8 violations, unused imports, and insecure code practices.",
    "suggest_improvements": "Suggest improvements for this code to enhance readability, maintainability, and efficiency.",
}


async def process_and_analyze_file(file, assistance_type=None, save_to_disk=False):
    try:
        logger.info(f"Starting analysis for file: {file.filename}")

        # Validate assistance type
        if assistance_type and assistance_type not in ASSISTANCE_OPTIONS:
            raise ValueError(f"Invalid assistance type: {assistance_type}")

        # Read file content
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes from the file.")

        # Optionally save file to disk
        if save_to_disk:
            temp_dir = Path(__file__).resolve().parents[2] / "debug"
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / file.filename
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            logger.info(f"File saved to {temp_file_path}")

        # Handle ZIP extraction
        project_structure = ""
        combined_files = ""
        if file.filename.endswith(".zip"):
            with zipfile.ZipFile(file.file, "r") as zip_ref:
                file_contents = {
                    file_name: zip_ref.read(file_name).decode("utf-8")
                    for file_name in zip_ref.namelist()
                }
                project_structure = "\n".join([f"- {name}" for name in file_contents.keys()])
                combined_files = "\n\n".join(
                    [f"### {name}\n\n{content}" for name, content in file_contents.items()]
                )
            logger.info(f"Extracted project structure:\n{project_structure}")

        # Prepare ChatGPT messages
        messages = [
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

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=5,
            temperature=0.7,
        )

        # Parse response
        result = response.choices[0].message.content.strip()
        logger.info(f"OpenAI response: {result}")

        return {
            "filename": file.filename,
            "analysis": {"summary": result},  # Ensure 'analysis' is a dictionary
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        return {
            "error": "An error occurred during processing.",
            "details": str(e),
            "success": False,
        }
