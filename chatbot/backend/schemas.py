# backend/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AgentResponse(BaseModel):
    answer: str = Field(..., description="Final response text after processing (summarized or compared)")
    sources: List[str] = Field(default_factory=list, description="List of source URLs used for generating the response")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured version of the answer for UI rendering")
