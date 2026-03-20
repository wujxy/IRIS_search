"""
Web routes for IRIS Web module.
HTML page routes for literature browsing.
"""

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
from collections import Counter
from services.paper_service import PaperService
from web.dependencies import get_paper_service
from web.template_config import templates

router = APIRouter(tags=["web"])


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    page: int = 1,
    category: Optional[str] = None,
    order_by: str = "published",
    paper_service: PaperService = Depends(get_paper_service)
):
    """
    Literature list page with pagination.
    """
    per_page = 10
    offset = (page - 1) * per_page

    # Get papers - use search_papers when category filter is active
    if category:
        # For category filter, we need to get all results and paginate manually
        all_papers = paper_service.search_papers(category=category, limit=None)
        total = len(all_papers)
        papers = all_papers[offset:offset + per_page]
    else:
        # Use list_papers for normal pagination
        papers = paper_service.list_papers(
            limit=per_page,
            offset=offset,
            order_by=order_by,
            reverse=True
        )
        stats = paper_service.get_stats()
        total = stats['total']

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    # Get categories for filter
    categories = _get_categories_with_counts(paper_service)

    # Parse authors and categories for display
    for paper in papers:
        paper['authors_list'] = parse_authors(paper.get('authors'))
        paper['categories_list'] = parse_categories(paper.get('categories'))
        paper['summary_preview'] = truncate_text(paper.get('summary'), 300)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "papers": papers,
            "categories": categories,
            "current_category": category,
            "pagination": {
                "page": page,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
                "total": total
            }
        }
    )


@router.get("/paper/{paper_id}", response_class=HTMLResponse)
async def paper_detail(
    request: Request,
    paper_id: str,
    paper_service: PaperService = Depends(get_paper_service)
):
    """
    Paper detail page.
    """
    paper = paper_service.get_paper_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

    # Parse data for display
    paper['authors_list'] = parse_authors(paper.get('authors'))
    paper['categories_list'] = parse_categories(paper.get('categories'))

    # Get QA fields
    qa_fields = []
    for i in range(1, 6):
        qa_key = f'q{i}'
        if paper.get(qa_key):
            qa_fields.append({'q': f'Q{i}', 'a': paper[qa_key]})

    return templates.TemplateResponse(
        "detail.html",
        {
            "request": request,
            "paper": paper,
            "qa_fields": qa_fields
        }
    )


@router.get("/search", response_class=HTMLResponse)
async def search_page(
    request: Request,
    q: Optional[str] = None,
    category: Optional[str] = None,
    paper_service: PaperService = Depends(get_paper_service)
):
    """
    Search results page.
    """
    papers = []
    count = 0
    categories = _get_categories_with_counts(paper_service)

    if q or category:
        papers = paper_service.search_papers(
            keyword=q,
            category=category,
            limit=50
        )
        count = len(papers)

        # Parse for display
        for paper in papers:
            paper['authors_list'] = parse_authors(paper.get('authors'))
            paper['categories_list'] = parse_categories(paper.get('categories'))
            paper['summary_preview'] = truncate_text(paper.get('summary'), 300)

    return templates.TemplateResponse(
        "index.html",  # Reuse index template for search results
        {
            "request": request,
            "papers": papers,
            "categories": categories,
            "current_category": category,
            "search_query": q,
            "is_search": True,
            "search_count": count,
            "pagination": {
                "page": 1,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False,
                "total": count
            }
        }
    )


def _get_categories_with_counts(paper_service: PaperService) -> list:
    """Get all categories with paper counts."""
    # Get all papers to extract categories
    papers = paper_service.list_papers(limit=None)
    categories = [p.get('primary_category') for p in papers if p.get('primary_category')]
    category_counts = Counter(categories)
    return [
        {'category': cat, 'count': count}
        for cat, count in category_counts.most_common()
    ]


def parse_authors(authors: list) -> list:
    """Parse authors field to list."""
    if not authors:
        return []
    if isinstance(authors, list):
        return authors
    return []


def parse_categories(categories: list) -> list:
    """Parse categories field to list."""
    if not categories:
        return []
    if isinstance(categories, list):
        return categories
    return []


def truncate_text(text: Optional[str], length: int) -> str:
    """Truncate text to specified length."""
    if not text:
        return ""
    if len(text) <= length:
        return text
    return text[:length] + "..."
