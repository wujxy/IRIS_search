"""
QA API routes for IRIS Web module.
Handles question answering with conversation support.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from web.dependencies import get_qa_service
from web.models import (
    QARequest, QAResponse, ConversationCreateResponse,
    ConversationHistoryResponse, Message
)

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/conversation", response_model=ConversationCreateResponse)
async def create_conversation(qa_service=Depends(get_qa_service)):
    """Create a new conversation session."""
    session_id = qa_service.create_conversation()
    return ConversationCreateResponse(
        session_id=session_id,
        message="New conversation created"
    )


@router.post("/conversation/{session_id}", response_model=QAResponse)
async def query_conversation(
    session_id: str,
    request: QARequest,
    qa_service=Depends(get_qa_service)
):
    """
    Ask a question within a conversation session.
    Supports multi-turn dialogue with context.
    """
    # Validate session exists
    if session_id not in qa_service.list_conversations():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Validate specific mode requirements
    if request.mode == "specific" and not request.paper_id:
        raise HTTPException(
            status_code=400,
            detail="paper_id is required for specific mode"
        )

    try:
        answer = await qa_service.query_with_conversation(
            session_id=session_id,
            question=request.question,
            mode=request.mode,
            paper_id=request.paper_id,
            top_k=request.top_k
        )

        return QAResponse(
            answer=answer,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/conversation/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation(
    session_id: str,
    qa_service=Depends(get_qa_service)
):
    """Get conversation history for a session."""
    # Validate session exists
    if session_id not in qa_service.list_conversations():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    history = qa_service.get_conversation(session_id)

    # Convert to Message models
    messages = [
        Message(role=msg.get("role", "user"), content=msg.get("content", ""))
        for msg in history
    ]

    return ConversationHistoryResponse(
        session_id=session_id,
        messages=messages,
        mode="global"  # Default mode, could be stored with session
    )


@router.delete("/conversation/{session_id}")
async def delete_conversation(
    session_id: str,
    qa_service=Depends(get_qa_service)
):
    """Delete a conversation session."""
    # Validate session exists
    if session_id not in qa_service.list_conversations():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    qa_service.delete_conversation(session_id)

    return {
        "message": f"Conversation {session_id} deleted",
        "session_id": session_id
    }


@router.get("/conversations")
async def list_conversations(qa_service=Depends(get_qa_service)):
    """List all active conversation sessions."""
    sessions = qa_service.list_conversations()
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@router.post("/query", response_model=QAResponse)
async def single_query(
    request: QARequest,
    qa_service=Depends(get_qa_service)
):
    """
    Single query without conversation (stateless).
    Creates a temporary session for the query.
    """
    # Validate specific mode requirements
    if request.mode == "specific" and not request.paper_id:
        raise HTTPException(
            status_code=400,
            detail="paper_id is required for specific mode"
        )

    try:
        # Create temporary session
        session_id = qa_service.create_conversation()

        # Execute query
        answer = await qa_service.query_with_conversation(
            session_id=session_id,
            question=request.question,
            mode=request.mode,
            paper_id=request.paper_id,
            top_k=request.top_k
        )

        # Clean up temporary session
        qa_service.delete_conversation(session_id)

        return QAResponse(
            answer=answer,
            session_id=""  # Empty for stateless queries
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )
