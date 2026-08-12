"""
Prompt Builder

This module is responsible for building the final prompt
that will be sent to the Language Model (LLM).
"""

from pathlib import Path

from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import logger


class PromptBuilder:
    """
    Build prompts for the Language Model.
    """

    def __init__(self) -> None:
        """
        Initialize the PromptBuilder.
        """

        self.system_prompt_path = (
            Path(settings.prompts_dir)
            / settings.system_prompt_file
        )

        logger.info("PromptBuilder initialized.")

    def load_system_prompt(self) -> str:
        """
        Load the system prompt from disk.

        Returns:
            str: System prompt.
        """

        try:

            system_prompt = self.system_prompt_path.read_text(
                encoding="utf-8"
            )

            logger.info("System prompt loaded successfully.")

            return system_prompt

        except Exception:

            logger.exception(
                "Failed to load system prompt."
            )

            raise

    def build_context(
        self,
        documents: list[Document]
    ) -> str:
        """
        Build formatted context from retrieved documents.

        Args:
            documents:
                Retrieved documents.

        Returns:
            Formatted context string.
        """

        try:

            contexts = []

            for index, document in enumerate(
                documents,
                start=1
            ):

                contexts.append(
                    f"""
Context {index}
{'=' * 80}

Task ID:
{document.metadata.get("task_id", "Unknown")}

Function:
{document.metadata.get("entry_point", "Unknown")}

Content:

{document.page_content}
""".strip()
                )

            context = "\n\n".join(contexts)

            logger.info(
                "Built context from %d documents.",
                len(documents)
            )

            return context

        except Exception:

            logger.exception(
                "Failed to build context."
            )

            raise

    def build_prompt(
        self,
        query: str,
        documents: list[Document]
    ) -> str:
        """
        Build the final prompt.

        Args:
            query:
                User question.

            documents:
                Retrieved documents.

        Returns:
            Final prompt.
        """

        try:

            system_prompt = self.load_system_prompt()

            context = self.build_context(documents)

            prompt = f"""
{system_prompt}

{'#' * 100}

Retrieved Context

{'#' * 100}

{context}

{'#' * 100}

User Question

{query}

{'#' * 100}

Answer:
""".strip()

            logger.info(
                "Prompt built successfully."
            )

            return prompt

        except Exception:

            logger.exception(
                "Failed to build prompt."
            )

            raise