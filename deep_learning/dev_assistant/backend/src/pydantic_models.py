from pydantic import BaseModel
from typing import Optional, Dict


class AnalysisResponse(BaseModel):
    filename: str
    analysis: Dict
    success: bool


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str]
    success: bool
