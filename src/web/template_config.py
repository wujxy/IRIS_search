"""
Template configuration for IRIS Web module.
Shared Jinja2Templates instance with custom filters.
"""

from fastapi.templating import Jinja2Templates

# Create shared templates instance
templates = Jinja2Templates(directory="src/web/templates")

# Add custom template filters
def truncate_filter(text: str, length: int = 100) -> str:
    """Truncate text to specified length."""
    if not text:
        return ""
    if len(text) <= length:
        return text
    return text[:length] + "..."

def format_date_filter(date_str: str) -> str:
    """Format ISO date string to readable format."""
    if not date_str:
        return "N/A"
    try:
        return date_str[:10]  # Extract YYYY-MM-DD
    except:
        return date_str

def author_list_filter(authors: list) -> str:
    """Format author list for display."""
    if not authors:
        return "Unknown"
    if len(authors) <= 3:
        return ", ".join(authors)
    return ", ".join(authors[:3]) + f" et al. ({len(authors)} authors)"

# Register filters
templates.env.filters["truncate"] = truncate_filter
templates.env.filters["format_date"] = format_date_filter
templates.env.filters["author_list"] = author_list_filter
