import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from src.config import USE_REAL_OPENAI_API, DUMMY_RESPONSE
from src.debug.zip_files import zip_files
from src.main import app

client = TestClient(app)


@pytest.fixture
def debug_zip_file():
    """
    Fixture to provide the debug zip file created by the `zip_files.py` script.
    """
    zip_path = Path(__file__).resolve().parents[1] / "debug" / "debug_files.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Debug zip file not found at {zip_path}. Please ensure the file exists or create it using `zip_files.py`."
        )
    return zip_path


@pytest.fixture
def project_structure_content():
    """
    Fixture providing the mandatory project structure content.
    """
    return """
    hands_on_keras/
    ├── README.md
    ├── __init__.py
    ├── artifacts/
    │   ├── final_stats/
    │   │   ├── all_metrics_example.html
    │   │   └── length_bucket_results_example.csv
    │   ├── logs/
    │   │   ├── log_example.log
    │   │   └── log_example.txt
    │   └── plots/
    │       └── EDA/
    ├── config.py
    └── main.py
    """


@pytest.fixture
def multi_file_zip(tmp_path):
    """
    Fixture to create a ZIP file with multiple files for testing.
    """
    zip_path = tmp_path / "multi_file.zip"
    files_to_zip = [
        tmp_path / "main.py",
        tmp_path / "utils.py",
        tmp_path / "tests/test_backend.py",
    ]
    # Create dummy files
    for file_path in files_to_zip:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(f"def dummy_func():\n    pass  # {file_path.name}")

    zip_files(zip_path, files_to_zip)
    return zip_path


# Helper function
def make_request(
    zip_file, project_structure_content, assistance_type="code_review", save_to_disk=False
):
    """
    Helper function to make a POST request to the /analyze/ endpoint.

    Args:
        zip_file: The main ZIP file to upload.
        project_structure_content: The content of project_structure.md.
        assistance_type (str): Type of developer assistance requested.
        save_to_disk (bool): Whether to enable saving to disk.

    Returns:
        Response object from the API.
    """
    return client.post(
        "/analyze/",
        files={
            "file": zip_file,
            "project_structure_file": (
                "project_structure.md",
                project_structure_content,
                "text/markdown",
            ),
        },
        params={"assistance_type": assistance_type, "save_to_disk": str(save_to_disk).lower()},
    )


# Tests
def test_analyze_debug_zip_in_memory(debug_zip_file, project_structure_content):
    """
    Test the `/analyze/` endpoint with in-memory processing and a mandatory project structure file.
    """
    with open(debug_zip_file, "rb") as zip_f:
        response = make_request(zip_f, project_structure_content)

    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "debug_files.zip", "Filename should match"

    if not USE_REAL_OPENAI_API:
        assert (
            json_data["analysis"]["summary"] == DUMMY_RESPONSE
        ), "Expected dummy response when USE_REAL_OPENAI_API is False"


def test_analyze_debug_zip_save_to_disk(debug_zip_file, project_structure_content):
    """
    Test the `/analyze/` endpoint with saving to disk enabled.
    """
    with open(debug_zip_file, "rb") as zip_f:
        response = make_request(zip_f, project_structure_content, save_to_disk=True)

    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "debug_files.zip", "Filename should match"

    if not USE_REAL_OPENAI_API:
        assert (
            json_data["analysis"]["summary"] == DUMMY_RESPONSE
        ), "Expected dummy response when USE_REAL_OPENAI_API is False"


def test_analyze_multi_file_zip(multi_file_zip, project_structure_content):
    """
    Test the `/analyze/` endpoint with a multi-file ZIP.
    """
    with open(multi_file_zip, "rb") as zip_f:
        response = make_request(zip_f, project_structure_content)

    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "multi_file.zip", "Filename should match"

    if not USE_REAL_OPENAI_API:
        assert (
            json_data["analysis"]["summary"] == DUMMY_RESPONSE
        ), "Expected dummy response when USE_REAL_OPENAI_API is False"


def test_missing_project_structure(debug_zip_file):
    """
    Test that the API returns a 422 error if project_structure.md is missing.
    """
    with open(debug_zip_file, "rb") as zip_f:
        response = client.post(
            "/analyze/",
            files={"file": zip_f},
            params={"assistance_type": "code_review"},
        )
    assert response.status_code == 422, "Expected status code 422 for missing project_structure.md"
    json_data = response.json()
    assert "detail" in json_data, "Response should contain a detailed error message"
    assert any(
        "project_structure_file" in error.get("loc", []) for error in json_data.get("detail", [])
    ), "Expected error indicating missing project_structure_file"


def test_invalid_zip_file(tmp_path, project_structure_content):
    """
    Test that the API returns a 400 error if an invalid ZIP file is uploaded.
    """
    invalid_zip = tmp_path / "invalid.zip"
    invalid_zip.write_text("This is not a valid zip file.")

    with open(invalid_zip, "rb") as zip_f:
        response = make_request(zip_f, project_structure_content)

    assert response.status_code == 400, "Expected status code 400 for invalid ZIP"
    json_data = response.json()
    assert "detail" in json_data, "Response should contain a detailed error message"
    assert (
        json_data["detail"] == "Invalid ZIP file uploaded. Please provide a valid ZIP file."
    ), "Expected error message for invalid ZIP"
