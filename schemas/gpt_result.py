from typing import Optional, List

from pydantic import BaseModel


class GPTSource(BaseModel):
    key: str
    url: str
    title: str

class GPTResult(BaseModel):
    source_confidence: float
    content: Optional[str]
    used_sources: List[GPTSource]

