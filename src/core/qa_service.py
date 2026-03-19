"""
QA Service for IRIS
RAG-based question answering service independent of UltraRAG.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from core.retriever import Retriever

logger = logging.getLogger(__name__)


class QAService:
    """
    Question answering service using RAG with LLM generation.
    """

    def __init__(
        self,
        retriever: Retriever,
        base_url: str,
        model_name: str,
        system_prompt: str = "你是一个专业的问答助手。请一定记住使用中文回答问题，且足够专业。",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0
    ):
        """
        Initialize QA service.

        Args:
            retriever: Retriever instance
            base_url: vLLM generation server URL
            model_name: Model name for generation
            system_prompt: System prompt for the LLM
            temperature: Generation temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 2048)
            timeout: Request timeout (default: 120.0)
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai is not installed. Install it with `pip install openai`"
            )

        self.retriever = retriever
        self.base_url = base_url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="dummy",
            timeout=timeout
        )

        # Conversation storage (in-memory, can be replaced with persistent storage)
        self._conversations: Dict[str, List[Dict]] = {}

        logger.info(
            f"QA Service initialized: model={model_name}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

    async def query(
        self,
        question: str,
        chunks_path: Optional[Path] = None,
        mode: str = "global",
        paper_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        top_k: int = 5
    ) -> str:
        """
        Answer a question using RAG.

        Args:
            question: Question to answer
            chunks_path: Path to chunks file (deprecated, kept for compatibility)
            mode: Query mode ("global" or "specific")
            paper_id: Paper ID for specific mode
            conversation_history: Conversation history for context
            top_k: Number of retrieved chunks (default: 5)

        Returns:
            Answer string

        Raises:
            ValueError: If mode is invalid
        """
        logger.info(f"Querying: '{question}', mode={mode}")

        # 1. Retrieve relevant chunks
        retrieved = await self.retriever.retrieve(
            query=question,
            mode=mode,
            paper_id=paper_id,
            top_k=top_k
        )

        if not retrieved:
            logger.warning("No relevant chunks retrieved")
            return "抱歉，没有找到相关的文档内容。"

        # 2. Build context
        context = self._build_context(retrieved)

        # 3. Build messages
        messages = self._build_messages(
            question=question,
            context=context,
            conversation_history=conversation_history
        )

        # 4. Generate answer
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content
            logger.info(f"Generated answer: {len(answer)} chars")
            return answer

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"抱歉，生成答案时出错: {str(e)}"

    async def query_knowledge_base(
        self,
        question: str,
        chunks_path: Path,
        mode: str = "global",
        paper_id: Optional[str] = None
    ) -> str:
        """
        Query knowledge base (backward compatibility method).

        Args:
            question: Question to answer
            chunks_path: Path to chunks file (unused, kept for compatibility)
            mode: Query mode
            paper_id: Paper ID for specific mode

        Returns:
            Answer string
        """
        return await self.query(
            question=question,
            chunks_path=chunks_path,
            mode=mode,
            paper_id=paper_id
        )

    async def query_knowledge_base_with_mode(
        self,
        questions: List[str],
        chunks_path: Optional[Path] = None,
        mode: str = "global",
        paper_id: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Query multiple questions (batch mode).

        Args:
            questions: List of questions
            chunks_path: Path to chunks file (deprecated)
            mode: Query mode
            paper_id: Paper ID for specific mode
            collection_name: Milvus collection name (unused, from retriever)

        Returns:
            List of {"question": str, "answer": str} dictionaries
        """
        results = []
        for question in questions:
            answer = await self.query(
                question=question,
                chunks_path=chunks_path,
                mode=mode,
                paper_id=paper_id
            )
            results.append({"question": question, "answer": answer})

        logger.info(f"Answered {len(results)} questions")
        return results

    def _build_context(self, retrieved: List[Dict]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            retrieved: List of retrieved chunk dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            title = chunk.get('title', 'Unknown')
            doc_id = chunk.get('doc_id', 'Unknown')
            content = chunk.get('contents', '')

            context_parts.append(
                f"[文档 {i}]\n"
                f"论文ID: {doc_id}\n"
                f"标题: {title}\n"
                f"内容: {content}"
            )

        return "\n\n".join(context_parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Build chat messages with system prompt and history.

        Args:
            question: Current question
            context: Retrieved context
            conversation_history: Optional conversation history

        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current question with context
        user_content = (
            "请参考以下文档内容回答问题。如果文档中没有相关信息，请诚实说明。\n\n"
            f"文档内容：\n{context}\n\n"
            f"问题：{question}"
        )
        messages.append({"role": "user", "content": user_content})

        return messages

    # Multi-turn conversation support
    def create_conversation(self) -> str:
        """
        Create a new conversation session.

        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid.uuid4())
        self._conversations[session_id] = []
        logger.debug(f"Created conversation: {session_id}")
        return session_id

    async def query_with_conversation(
        self,
        session_id: str,
        question: str,
        mode: str = "global",
        paper_id: Optional[str] = None,
        top_k: int = 5
    ) -> str:
        """
        Query with conversation context.

        Args:
            session_id: Conversation session ID
            question: Current question
            mode: Query mode
            paper_id: Paper ID for specific mode
            top_k: Number of retrieved chunks

        Returns:
            Answer string
        """
        # Get conversation history
        history = self._conversations.get(session_id, [])

        # Query with history
        answer = await self.query(
            question=question,
            mode=mode,
            paper_id=paper_id,
            conversation_history=history,
            top_k=top_k
        )

        # Save to history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        # Limit history to last 10 messages to avoid context overflow
        if len(history) > 20:  # 10 turns
            history = history[-20:]
            self._conversations[session_id] = history

        return answer

    def get_conversation_history(
        self,
        session_id: str
    ) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID

        Returns:
            List of message dictionaries
        """
        return self._conversations.get(session_id, [])

    def delete_conversation(self, session_id: str) -> bool:
        """
        Delete a conversation session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.debug(f"Deleted conversation: {session_id}")
            return True
        return False

    def list_conversations(self) -> List[str]:
        """
        List all active conversation session IDs.

        Returns:
            List of session IDs
        """
        return list(self._conversations.keys())

    def close(self) -> None:
        """Close the client connection."""
        if self.client:
            self.client.close()
        logger.debug("QA client closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_qa_service_from_config(config: dict) -> QAService:
    """
    Factory function to create QAService from configuration.

    Args:
        config: Configuration dictionary with the following keys:
            - embedding.base_url
            - embedding.model_name
            - embedding.batch_size (optional)
            - milvus.uri
            - milvus.collection_name
            - milvus.embedding_dim
            - milvus.token (optional)
            - reranker.enabled (optional)
            - reranker.model_path (optional)
            - reranker.device (optional)
            - reranker.batch_size (optional)
            - retrieval.top_k (optional)
            - retrieval.rerank_multiplier (optional)
            - qa.base_url (generation server URL)
            - qa.system_prompt (optional)
            - qa.temperature (optional)
            - qa.max_tokens (optional)

    Returns:
        Configured QAService instance
    """
    # Create retriever first
    from core.retriever import create_retriever_from_config
    retriever = create_retriever_from_config(config)

    # QA configuration
    qa_config = config.get("qa", {})
    # Extract model name from qa.model_name or llm_model_path
    llm_path = config.get("models", {}).get("llm_model_path", "")
    default_model_name = Path(llm_path).name if llm_path else "llama-3-2-3b-instruct"
    model_name = qa_config.get("model_name", default_model_name)

    # Create QA service
    qa_service = QAService(
        retriever=retriever,
        base_url=qa_config.get("base_url", "http://127.0.0.1:65504/v1"),
        model_name=model_name,
        system_prompt=qa_config.get("system_prompt", "你是一个专业的文献问答助手。请一定记住使用中文回答问题，且足够专业。"),
        temperature=qa_config.get("temperature", 0.7),
        max_tokens=qa_config.get("max_tokens", 2048),
        timeout=qa_config.get("timeout", 120.0)
    )

    logger.info("QAService created from config")
    return qa_service
