from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from logging_config import setup_logger
from services import process_and_analyze_file
from pydantic_models import AnalysisResponse, ErrorResponse

logger = setup_logger(__name__)

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete.")


@app.post(
    "/analyze/",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_file(
    file: UploadFile = File(...),
    save_to_disk: bool = False,
    assistance_type: str = Query(..., description="Type of developer assistance requested"),
):
    """
    Endpoint to handle file uploads and perform analysis.

    Args:
        file (UploadFile): The uploaded file.
        save_to_disk (bool): Optional; Whether to save the file to disk before processing.
        assistance_type (str): The type of developer assistance requested.

    Returns:
        AnalysisResponse: The analysis result.
    """
    try:
        # Validate the file type
        if not file.filename.endswith(".zip"):
            raise HTTPException(
                status_code=400,
                detail="Only .zip files are supported.",
            )

        # Process the file
        result = await process_and_analyze_file(file, assistance_type, save_to_disk=save_to_disk)
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Unknown error occurred."),
            )

        return result
    except HTTPException as e:
        logger.warning(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the file.",
        )
