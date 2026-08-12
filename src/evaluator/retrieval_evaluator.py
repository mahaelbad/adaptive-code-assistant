"""
Retrieval Evaluator

This module evaluates whether retrieved documents are relevant
to the user's code-generation request.
"""

from langchain_core.documents import Document

from src.llm.llm_client import LLMClient
from src.utils.logger import logger


class RetrievalEvaluator:
    """
    Uses an LLM to evaluate whether retrieved documents
    are relevant to the user's request.
    """

    def __init__(self) -> None:
        """
        Initialize the Retrieval Evaluator.
        """

        self.llm = LLMClient(
            temperature=0.0,
            max_tokens=20,
        )

        logger.info(
            "Retrieval Evaluator initialized successfully."
        )

    def _build_prompt(
        self,
        query: str,
        documents: list[Document],
    ) -> str:
        """
        Build a compact prompt for retrieval evaluation.
        """

        context_parts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            task_id = document.metadata.get(
                "task_id",
                "Unknown",
            )

            entry_point = document.metadata.get(
                "entry_point",
                "Unknown",
            )

            content = document.page_content.strip()

            # Keep the evaluator prompt small.
            if len(content) > 1000:
                content = content[:1000]

            context_parts.append(
                f"""
Example {index}
Task ID: {task_id}
Function: {entry_point}
Content:
{content}
""".strip()
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a binary retrieval evaluator.

User request:
{query}

Retrieved examples:
{context}

Decide whether at least one retrieved example
is useful for solving the user's request.

Useful means it contains a similar programming task,
algorithm, data structure, filtering, sorting,
iteration, or Python programming pattern.

Return ONLY one of these exact labels:

RELEVANT
NOT_RELEVANT

Do not explain.
Do not use markdown.
Do not add punctuation.

Answer:
""".strip()

        return prompt

    def evaluate(
        self,
        query: str,
        documents: list[Document],
    ) -> bool:
        """
        Evaluate whether retrieved documents are relevant.

        Returns:
            True if relevant.
            False if not relevant or evaluation fails.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            logger.warning(
                "No documents available for retrieval evaluation."
            )

            return False

        logger.info(
            "Evaluating %d retrieved documents.",
            len(documents),
        )

        prompt = self._build_prompt(
            query=query,
            documents=documents,
        )

        try:
            logger.info(
                "Sending retrieval evaluation request to LLM."
            )

            response = self.llm.generate(
                prompt
            )

            logger.info(
                "Raw retrieval evaluation response: %r",
                response,
            )

            if not response or not response.strip():
                logger.warning(
                    "Retrieval evaluator returned an empty response."
                )

                return False

            result = (
                response
                .strip()
                .upper()
                .replace("`", "")
                .strip()
            )

            # Exact matching is safer than
            # checking "in" because:
            # NOT_RELEVANT contains RELEVANT.
            if result == "RELEVANT":
                logger.info(
                    "Retrieved context is relevant."
                )

                return True

            if result == "NOT_RELEVANT":
                logger.info(
                    "Retrieved context is not relevant."
                )

                return False

            # Sometimes the model may return extra text.
            # Look for the complete label first.
            if "NOT_RELEVANT" in result:
                logger.warning(
                    "Non-standard evaluator response detected: %r",
                    response,
                )

                return False

            if "RELEVANT" in result:
                logger.warning(
                    "Non-standard evaluator response detected: %r",
                    response,
                )

                return True

            logger.warning(
                "Invalid retrieval evaluation response: %r",
                response,
            )

            return False

        except Exception:
            logger.exception(
                "Retrieval evaluation failed. "
                "Falling back to NOT_RELEVANT."
            )

            return False