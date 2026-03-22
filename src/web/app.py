"""
FastAPI application for IRIS Web module.
Main application entry point.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

from web.routers import web_routes, api_routes, qa_routes
from web.template_config import templates
from web.dependencies import ModelUnavailableError

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="IRIS Literature Browser",
    description="Web interface for IRIS research paper database",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Include routers
app.include_router(web_routes.router)
app.include_router(api_routes.router)
app.include_router(qa_routes.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "iris-web"}


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    if exc.status_code == 404:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "status_code": 404, "message": "Page not found"},
            status_code=404
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )


@app.exception_handler(ModelUnavailableError)
async def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
    """Handle model unavailable errors with bilingual messages."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "model_unavailable",
            "message_en": exc.message_en,
            "message_zh": exc.message_zh,
            "model_type": exc.model_type
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "status_code": 500, "message": "Internal server error"},
        status_code=500
    )


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("IRIS Web application starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("IRIS Web application shutting down...")
