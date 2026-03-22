"""
QA Service for IRIS
RAG-based question answering service supporting local vLLM and external APIs.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.retriever import Retriever

# Optional provider imports
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    import cohere
except ImportError:
    cohere = None

logger = logging.getLogger(__name__)


class QAService:
    """
    Universal QA service supporting local vLLM and external chat APIs.

    Supported providers:
    - local: vLLM server (default)
    - openai: OpenAI chat API
    - anthropic: Anthropic Claude API
    - cohere: Cohere chat API
    """

    def __init__(
        self,
        retriever: Retriever,
        base_url: str,
        model_name: str,
        provider: str = "local",
        api_key: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        system_prompt: str = "你是一个专业的问答助手。请一定记住使用中文回答问题，且足够专业。",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0
    ):
        """
        Initialize QA service with provider detection.

        Args:
            retriever: Retriever instance
            base_url: API base URL
            model_name: Model name for generation
            provider: Provider type (local, openai, anthropic, cohere)
            api_key: API key for external providers
            provider_config: Provider-specific configuration
            system_prompt: System prompt for the LLM
            temperature: Generation temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 2048)
            timeout: Request timeout (default: 120.0)
        """
        self.provider = provider or "local"
        self.provider_config = provider_config or {}
        self.retriever = retriever
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Validate and initialize provider
        if self.provider in ["local", "openai"]:
            self._init_openai_client(base_url, model_name, api_key, timeout)
            self._generate_method = self._generate_openai
        elif self.provider == "anthropic":
            self._init_anthropic_client(api_key, model_name)
            self._generate_method = self._generate_anthropic
        elif self.provider == "cohere":
            self._init_cohere_client(api_key, model_name)
            self._generate_method = self._generate_cohere
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.model_name = model_name

        # Conversation storage (in-memory, can be replaced with persistent storage)
        self._conversations: Dict[str, List[Dict]] = {}

        logger.info(
            f"QA Service initialized ({self.provider}): model={model_name}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

    def _init_openai_client(self, base_url: str, model_name: str, api_key: Optional[str], timeout: float):
        """Initialize OpenAI-compatible client for local and OpenAI."""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai is not installed. Install it with `pip install openai`"
            )

        # For local provider, use dummy key; for OpenAI, require real key
        if self.provider == "openai" and not api_key:
            raise ValueError("api_key is required for openai provider")

        openai_config = self.provider_config.get("openai", {})
        final_model_name = openai_config.get("model", model_name)

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "dummy",
            timeout=timeout
        )
        self.model_name = final_model_name

    def _init_anthropic_client(self, api_key: Optional[str], model_name: str):
        """Initialize Anthropic Claude client."""
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic is not installed. Install it with `pip install anthropic`"
            )
        if not api_key:
            raise ValueError("api_key is required for anthropic provider")

        anthropic_config = self.provider_config.get("anthropic", {})

        self.client = AsyncAnthropic(api_key=api_key)
        self.model_name = anthropic_config.get("model", model_name)
        self.max_tokens = anthropic_config.get("max_tokens", self.max_tokens)
        self.anthropic_version = anthropic_config.get("version", "2023-05-22")

    def _init_cohere_client(self, api_key: Optional[str], model_name: str):
        """Initialize Cohere chat client."""
        if cohere is None:
            raise ImportError(
                "cohere is not installed. Install it with `pip install cohere`"
            )
        if not api_key:
            raise ValueError("api_key is required for cohere provider")

        cohere_config = self.provider_config.get("cohere", {})

        connect_timeout = cohere_config.get("connect_timeout", self.timeout)
        self.client = cohere.AsyncClient(api_key, timeout=connect_timeout)
        self.model_name = cohere_config.get("model", model_name)

    async def _generate_openai(self, messages: List[Dict]) -> str:
        """Generate using OpenAI-compatible API (local vLLM or OpenAI)."""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    async def _generate_anthropic(self, messages: List[Dict]) -> str:
        """Generate using Anthropic Claude API."""
        # Extract system message from messages
        system_message = self.system_prompt
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                # Convert OpenAI format to Anthropic format
                user_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        response = await self.client.messages.create(
            model=self.model_name,
            system=system_message,
            messages=user_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        # Anthropic returns content blocks
        return response.content[0].text

    async def _generate_cohere(self, messages: List[Dict]) -> str:
        """Generate using Cohere chat API."""
        # Extract current message and chat history
        current_message = messages[-1]["content"] if messages else ""

        # Build chat history (excluding system prompt and current message)
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] in ["user", "assistant"]:
                chat_history.append({
                    "role": msg["role"],
                    "message": msg["content"]
                })

        response = await self.client.chat(
            model=self.model_name,
            message=current_message,
            chat_history=chat_history if chat_history else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.text

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
        logger.info(f"Querying: '{question}', mode={mode}, provider={self.provider}")

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

        # 4. Generate answer using provider-specific method
        try:
            answer = await self._generate_method(messages)
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
        logger.debug(f"Building context from {len(retrieved)} chunks")

        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            # Debug: log available fields
            logger.debug(f"Chunk {i} keys: {list(chunk.keys())}")

            title = chunk.get('title', 'Unknown')
            doc_id = chunk.get('doc_id', 'Unknown')
            # Try multiple possible field names for content
            content = (
                chunk.get('contents') or
                chunk.get('content') or
                chunk.get('text') or
                ''
            )

            # Debug: log content length
            logger.debug(f"Chunk {i} contents length: {len(str(content))}, doc_id: {doc_id}")

            # If content is still empty, warn and skip
            if not content:
                logger.warning(f"Chunk {i} has empty content, doc_id: {doc_id}")
                continue

            context_parts.append(
                f"[文档 {i}]\n"
                f"论文ID: {doc_id}\n"
                f"标题: {title}\n"
                f"内容: {content}"
            )

        result = "\n\n".join(context_parts)
        logger.debug(f"Built context with {len(context_parts)} valid chunks, total length: {len(result)}")
        return result

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
        if hasattr(self, 'client') and self.client:
            if self.provider == "anthropic":
                # Anthropic client doesn't have close method
                pass
            else:
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
            - models.reranker_model_path
            - reranker.device (optional)
            - reranker.batch_size (optional)
            - retrieval.top_k (optional)
            - retrieval.rerank_multiplier (optional)
            - qa.base_url (generation server URL)
            - qa.provider (optional, default: "local")
            - qa.api_key (optional, for external providers)
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
        provider=qa_config.get("provider", "local"),
        api_key=qa_config.get("api_key"),
        provider_config=qa_config,
        system_prompt=qa_config.get("system_prompt", "你是一个专业的文献问答助手。请一定记住使用中文回答问题，且足够专业。"),
        temperature=qa_config.get("temperature", 0.7),
        max_tokens=qa_config.get("max_tokens", 2048),
        timeout=qa_config.get("timeout", 120.0)
    )

    logger.info("QAService created from config")
    return qa_service
