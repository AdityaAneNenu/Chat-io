from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AgentResponse(BaseModel):
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Formatted response text")
    sources: List[str] = Field(
        default_factory=list,
        description="List of source URLs or references"
    )
    structured_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured version of the response"
    )
    format_type: str = Field(
        default="markdown",
        description="Format of the answer (markdown/html/plain)"
    )