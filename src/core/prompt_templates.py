"""
Prompt Templates for IRIS
Jinja2-based template management for LLM prompts.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
except ImportError:
    Environment = None
    Template = None

logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    Template manager using Jinja2 for LLM prompt construction.
    """

    def __init__(
        self,
        template_content: str = None,
        template_path: Optional[Path] = None,
        template_dir: Optional[Path] = None
    ):
        """
        Initialize prompt template.

        Args:
            template_content: Template string (if provided)
            template_path: Path to template file
            template_dir: Directory containing templates (for file-based templates)

        Note:
            Provide either template_content or template_path, not both.
        """
        if Environment is None:
            raise ImportError(
                "jinja2 is not installed. Install it with `pip install jinja2`"
            )

        if template_content and template_path:
            raise ValueError("Provide either template_content or template_path, not both")

        self.template_content = template_content
        self.template_path = template_path
        self.template_dir = template_dir

        # Load template
        if template_path:
            self._load_from_file(template_path, template_dir)
        elif template_content:
            self.template = Template(template_content, undefined=StrictUndefined)
            logger.debug("Template loaded from content string")
        else:
            raise ValueError("Either template_content or template_path must be provided")

    def _load_from_file(self, template_path: Path, template_dir: Optional[Path] = None):
        """
        Load template from file.

        Args:
            template_path: Path to template file
            template_dir: Base directory for templates
        """
        template_path = Path(template_path)

        if template_dir:
            template_dir = Path(template_dir)
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                undefined=StrictUndefined,
                autoescape=True
            )
            # Use relative path from template_dir
            rel_path = template_path.relative_to(template_dir)
            self.template = env.get_template(str(rel_path))
        else:
            # Load file content directly
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.template = Template(content, undefined=StrictUndefined)

        logger.debug(f"Template loaded from file: {template_path}")

    def render(self, **kwargs) -> str:
        """
        Render template with provided variables.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered string

        Raises:
            TypeError: If required variables are missing
        """
        try:
            return self.template.render(**kwargs)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise

    def validate_variables(self, **kwargs) -> Dict[str, bool]:
        """
        Validate if all required template variables are provided.

        Args:
            **kwargs: Variables to validate

        Returns:
            Dictionary of variable name -> present (bool)
        """
        if hasattr(self.template, 'environment'):
            # For Jinja2 templates loaded from files
            variables = self.template.environment.undefined(
                self.template
            ) if self.template.module.__dict__ else {}
        else:
            # For simple templates, get from code
            from jinja2 import meta
            ast = self.template.environment.parse(self.template.source)
            variables = meta.find_undeclared_variables(ast)

        return {v: v in kwargs for v in variables}

    @staticmethod
    def from_file(template_path: Path, template_dir: Optional[Path] = None) -> 'PromptTemplate':
        """
        Create PromptTemplate from file.

        Args:
            template_path: Path to template file
            template_dir: Base directory for templates

        Returns:
            PromptTemplate instance
        """
        return PromptTemplate(template_path=template_path, template_dir=template_dir)

    @staticmethod
    def from_string(template_content: str) -> 'PromptTemplate':
        """
        Create PromptTemplate from content string.

        Args:
            template_content: Template content string

        Returns:
            PromptTemplate instance
        """
        return PromptTemplate(template_content=template_content)


# Predefined templates
RAG_QA_TEMPLATE = """请参考以下文档内容回答问题。如果文档中没有相关信息，请诚实说明。

文档内容：
{% for chunk in chunks %}
[文档 {{ loop.index }}]
论文ID: {{ chunk.doc_id }}
标题: {{ chunk.title }}
内容: {{ chunk.contents }}

{% endfor %}

问题：{{ question }}

请使用中文回答，保持专业和准确。"""

SUMMARY_TEMPLATE = """请根据以下论文内容生成摘要。

论文ID: {{ paper_id }}
标题：{{ title }}
摘要：{{ summary }}

请生成以下内容：
1. 主要问题
2. 关键贡献
3. 使用方法
4. 重要概念
5. 研究方向

请使用中文回答。"""


class RAGQAPrompt:
    """
    Pre-configured RAG QA prompt template.
    """

    def __init__(self, system_prompt: str = None):
        """
        Initialize RAG QA prompt.

        Args:
            system_prompt: System prompt for the LLM
        """
        self.system_prompt = system_prompt or (
            "你是一个专业的文献问答助手。"
            "请使用中文回答问题，回答要准确、专业。"
        )
        self.qa_template = PromptTemplate.from_string(RAG_QA_TEMPLATE)

    def build_messages(
        self,
        question: str,
        chunks: list,
        conversation_history: list = None
    ) -> list:
        """
        Build chat messages for RAG QA.

        Args:
            question: Current question
            chunks: Retrieved chunks
            conversation_history: Optional conversation history

        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        # Render user prompt
        user_prompt = self.qa_template.render(
            question=question,
            chunks=chunks
        )

        messages.append({"role": "user", "content": user_prompt})

        return messages
