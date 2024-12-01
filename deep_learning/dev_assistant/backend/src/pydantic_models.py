from pydantic import BaseModel
from typing import Optional, Dict, Any


class AnalysisResponse(BaseModel):
    filename: str
    analysis: Dict[str, Any]  # Adjusted to accept any valid dictionary structure
    success: bool


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str]
    success: bool
