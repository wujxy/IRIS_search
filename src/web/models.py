"""
Pydantic models for IRIS Web API.
Defines request and response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PaperResponse(BaseModel):
    """Paper model for API responses."""
    id: int
    paper_id: str
    title: str
    authors: str
    published: Optional[str] = None
    summary: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    primary_category: Optional[str] = None
    categories: Optional[str] = None
    q1: Optional[str] = None
    q2: Optional[str] = None
    q3: Optional[str] = None
    q4: Optional[str] = None
    q5: Optional[str] = None
    pdf_url: Optional[str] = None

    class Config:
        from_attributes = True


class PaginatedPaperResponse(BaseModel):
    """Paginated response for paper list."""
    papers: List[PaperResponse]
    total: int
    page: int
    per_page: int
    total_pages: int
    has_next: bool
    has_prev: bool


class SearchRequest(BaseModel):
    """Search request model."""
    keyword: Optional[str] = None
    category: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)


class SearchResponse(BaseModel):
    """Search response model."""
    papers: List[PaperResponse]
    count: int


class CategoryResponse(BaseModel):
    """Category with count."""
    category: str
    count: int


class StatsResponse(BaseModel):
    """Database statistics."""
    total_papers: int
    categories: List[CategoryResponse]
    latest_paper: Optional[PaperResponse] = None


# ==================== QA Models ====================

class Message(BaseModel):
    """Chat message."""
    role: str  # "user", "assistant", "system"
    content: str


class QARequest(BaseModel):
    """QA query request."""
    question: str = Field(..., min_length=1)
    mode: str = Field("global", pattern="^(global|specific)$")
    paper_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


class QAResponse(BaseModel):
    """QA query response."""
    answer: str
    session_id: str


class ConversationCreateResponse(BaseModel):
    """Response when creating a new conversation."""
    session_id: str
    message: str = "New conversation created"


class ConversationHistoryResponse(BaseModel):
    """Conversation history response."""
    session_id: str
    messages: List[Message]
    mode: str = "global"
    paper_id: Optional[str] = None
