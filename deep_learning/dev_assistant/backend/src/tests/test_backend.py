from pathlib import Path
import pytest
import zipfile
from fastapi.testclient import TestClient
from config import USE_REAL_OPENAI_API, DUMMY_RESPONSE
from main import app

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


def test_analyze_debug_zip_in_memory(debug_zip_file):
    """
    Test the `/analyze/` endpoint with in-memory processing.
    """
    with open(debug_zip_file, "rb") as f:
        response = client.post(
            "/analyze/",
            files={"file": f},
            params={"assistance_type": "code_review"},
        )
    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "debug_files.zip", "Filename should match"

    if not USE_REAL_OPENAI_API:
        assert (
            json_data["analysis"]["summary"] == DUMMY_RESPONSE
        ), "Expected dummy response when USE_REAL_OPENAI_API is False"


def test_analyze_debug_zip_save_to_disk(debug_zip_file):
    """
    Test the `/analyze/` endpoint with saving to disk enabled.
    """
    with open(debug_zip_file, "rb") as f:
        response = client.post(
            "/analyze/",
            files={"file": f},
            params={"assistance_type": "code_review", "save_to_disk": "true"},
        )
    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "debug_files.zip", "Filename should match"

    if not USE_REAL_OPENAI_API:
        assert (
            json_data["analysis"]["summary"] == DUMMY_RESPONSE
        ), "Expected dummy response when USE_REAL_OPENAI_API is False"


@pytest.fixture
def multi_file_zip(tmp_path):
    """
    Fixture to create a ZIP file with multiple files for testing.
    """
    zip_path = tmp_path / "multi_file.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.writestr("main.py", "def main():\n    print('Hello, World!')")
        zipf.writestr("utils.py", "def util():\n    return True")
        zipf.writestr("tests/test_backend.py", "def test_backend():\n    assert True")
    return zip_path


def test_analyze_multi_file_zip(multi_file_zip):
    """
    Test the `/analyze/` endpoint with a multi-file ZIP.
    """
    with open(multi_file_zip, "rb") as f:
        response = client.post(
            "/analyze/",
            files={"file": f},
            params={"assistance_type": "code_review"},
        )
    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "multi_file.zip", "Filename should match"

    if not USE_REAL_OPENAI_API:
        assert (
            json_data["analysis"]["summary"] == DUMMY_RESPONSE
        ), "Expected dummy response when USE_REAL_OPENAI_API is False"

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
