from pathlib import Path
import pytest
import zipfile
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture
def debug_zip_file():
    """
    Fixture to provide the debug zip file created by the `zip_files.py` script.
    """
    zip_path = Path(__file__).resolve().parents[1] / "debug" / "debug_files.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Debug zip file not found at {zip_path}")
    return zip_path


def test_analyze_debug_zip_in_memory(debug_zip_file):
    """
    Test the `/analyze/` endpoint with in-memory processing.
    """
    with open(debug_zip_file, "rb") as f:
        response = client.post(
            "/analyze/", files={"file": f}, params={"assistance_type": "code_review"}
        )
    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "debug_files.zip", "Filename should match"


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
            "/analyze/", files={"file": f}, params={"assistance_type": "code_review"}
        )
    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "multi_file.zip", "Filename should match"
