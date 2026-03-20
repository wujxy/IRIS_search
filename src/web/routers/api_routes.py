"""
API routes for IRIS Web module.
JSON endpoints for paper data and search.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from collections import Counter
from services.paper_service import PaperService
from web.models import PaperResponse, PaginatedPaperResponse, SearchRequest, SearchResponse, CategoryResponse, StatsResponse
from web.dependencies import get_paper_service

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/papers", response_model=PaginatedPaperResponse)
async def get_papers(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    order_by: str = Query("published"),
    reverse: bool = Query(True),
    category: Optional[str] = None,
    paper_service: PaperService = Depends(get_paper_service)
):
    """
    Get paginated list of papers.

    - **page**: Page number (starts from 1)
    - **per_page**: Papers per page (max 100)
    - **order_by**: Field to sort by
    - **reverse**: Sort descending if True
    - **category**: Filter by primary_category
    """
    offset = (page - 1) * per_page

    # Get papers with filters
    if category:
        # For category filter, get all and paginate manually
        all_papers = paper_service.search_papers(category=category, limit=None)
        total = len(all_papers)
        papers = all_papers[offset:offset + per_page]
    else:
        # Use list_papers for normal pagination
        papers = paper_service.list_papers(
            limit=per_page,
            offset=offset,
            order_by=order_by,
            reverse=reverse
        )
        stats = paper_service.get_stats()
        total = stats['total']

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    # Convert to response models
    paper_responses = [PaperResponse(**dict(p)) for p in papers]

    return PaginatedPaperResponse(
        papers=paper_responses,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )


@router.get("/papers/{paper_id}", response_model=PaperResponse)
async def get_paper(
    paper_id: str,
    paper_service: PaperService = Depends(get_paper_service)
):
    """Get paper by ID (e.g., '2403.15570')."""
    paper = paper_service.get_paper_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
    return PaperResponse(**dict(paper))


@router.post("/search", response_model=SearchResponse)
async def search_papers(
    request: SearchRequest,
    paper_service: PaperService = Depends(get_paper_service)
):
    """
    Search papers by keyword and/or category.

    - **keyword**: Search in title, authors, summary
    - **category**: Filter by primary_category
    - **limit**: Maximum results to return
    """
    papers = paper_service.search_papers(
        keyword=request.keyword,
        category=request.category,
        limit=request.limit
    )

    paper_responses = [PaperResponse(**dict(p)) for p in papers]

    return SearchResponse(
        papers=paper_responses,
        count=len(paper_responses)
    )


@router.get("/categories", response_model=List[CategoryResponse])
async def get_categories(paper_service: PaperService = Depends(get_paper_service)):
    """Get all categories with paper counts."""
    # Get all papers to extract categories
    papers = paper_service.list_papers(limit=None)
    categories = [p.get('primary_category') for p in papers if p.get('primary_category')]
    category_counts = Counter(categories)

    return [
        CategoryResponse(category=cat, count=count)
        for cat, count in category_counts.most_common()
    ]


@router.get("/stats", response_model=StatsResponse)
async def get_stats(paper_service: PaperService = Depends(get_paper_service)):
    """Get database statistics."""
    stats = paper_service.get_stats()

    # Get categories
    papers = paper_service.list_papers(limit=None)
    categories = [p.get('primary_category') for p in papers if p.get('primary_category')]
    category_counts = Counter(categories)

    category_responses = [
        CategoryResponse(category=cat, count=count)
        for cat, count in category_counts.most_common()
    ]

    # Get latest paper
    latest = paper_service.list_papers(limit=1, order_by="published", reverse=True)
    latest_paper = None
    if latest:
        latest_paper = PaperResponse(**dict(latest[0]))

    return StatsResponse(
        total_papers=stats['total'],
        categories=category_responses,
        latest_paper=latest_paper
    )


@router.post("/query")
async def query_qa():
    """
    QA query endpoint - RESERVED FOR FUTURE IMPLEMENTATION.
    Will integrate with existing QA service.
    """
    raise HTTPException(status_code=501, detail="QA endpoint not yet implemented")
