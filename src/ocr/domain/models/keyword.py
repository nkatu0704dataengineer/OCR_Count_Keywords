from typing import List
from pydantic import BaseModel, Field


class KeywordCount(BaseModel):
    """Represents the count of a single keyword/phrase found in the document text."""
    keyword: str
    count: int = 0


class KeywordReport(BaseModel):
    """Aggregated keyword frequency report for an entire document."""
    document_id: str
    source_file: str
    keywords: List[KeywordCount] = Field(default_factory=list)
    total_keywords_searched: int = 0
    total_matches: int = 0
