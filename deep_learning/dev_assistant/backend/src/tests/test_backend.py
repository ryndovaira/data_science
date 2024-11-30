import pytest
from fastapi.testclient import TestClient
from main import app
from pathlib import Path

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
        response = client.post("/analyze/", files={"file": f})
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
        response = client.post("/analyze/?save_to_disk=true", files={"file": f})
    assert response.status_code == 200, "Expected status code 200"
    json_data = response.json()
    assert json_data["success"] is True, "Response should indicate success"
    assert "analysis" in json_data, "Response JSON should contain 'analysis'"
    assert json_data["filename"] == "debug_files.zip", "Filename should match"
