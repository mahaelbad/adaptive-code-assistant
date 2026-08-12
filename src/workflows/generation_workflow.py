"""
Code Generation Workflow

This module implements the complete code generation pipeline:

User Request
    ↓
Retriever
    ↓
Retrieval Evaluator
    ↓
Relevant?
    ├── Yes → RAG Prompt
    └── No  → Direct LLM Prompt
    ↓
LLM
    ↓
Final Response
"""

from langchain_core.documents import Document

from src.retriever.retriever import Retriever
from src.evaluator.retrieval_evaluator import RetrievalEvaluator
from src.llm.llm_client import LLMClient
from src.utils.logger import logger


class GenerationWorkflow:
    """
    Complete workflow for generating code solutions.

    The workflow first retrieves similar programming examples,
    evaluates their relevance, and then chooses between:

    1. RAG-based generation
    2. Direct LLM generation
    """

    def __init__(
        self,
        k: int = 3,
    ) -> None:
        """
        Initialize the generation workflow.

        Args:
            k:
                Number of documents to retrieve.
        """

        self.retriever = Retriever(
            k=k
        )

        self.evaluator = RetrievalEvaluator()

        self.llm = LLMClient(
            temperature=0.2,
            max_tokens=2000,
        )

        logger.info(
            "Generation Workflow initialized successfully."
        )

    def _build_rag_prompt(
        self,
        query: str,
        documents: list[Document],
    ) -> str:
        """
        Build a prompt using relevant retrieved documents.

        Args:
            query:
                User's programming request.

            documents:
                Retrieved relevant documents.

        Returns:
            RAG generation prompt.
        """

        context_parts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            context_parts.append(
                (
                    f"Retrieved Example {index}\n"
                    f"{'=' * 70}\n\n"
                    f"Task ID:\n"
                    f"{document.metadata.get('task_id', 'Unknown')}\n\n"
                    f"Function:\n"
                    f"{document.metadata.get('entry_point', 'Unknown')}\n\n"
                    f"Content:\n"
                    f"{document.page_content}"
                )
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = (
            "You are an expert Python programming assistant.\n\n"

            "The user wants a correct and complete solution "
            "to a programming problem.\n\n"

            "You have been provided with retrieved examples "
            "from a programming dataset.\n\n"

            "Use these examples only as supporting knowledge. "
            "Do not blindly copy them. Adapt the ideas to the "
            "user's actual requirements.\n\n"

            "RETRIEVED CONTEXT\n"
            + "=" * 80
            + "\n\n"
            + context
            + "\n\n"
            + "=" * 80
            + "\n\n"

            "USER REQUEST\n"
            + query
            + "\n\n"

            + "=" * 80
            + "\n\n"

            "INSTRUCTIONS\n\n"

            "1. Understand the user's requirements carefully.\n"
            "2. Generate a complete Python solution.\n"
            "3. Use the retrieved examples only when relevant.\n"
            "4. Do not copy unrelated code.\n"
            "5. Handle reasonable edge cases.\n"
            "6. Keep the code clean and readable.\n"
            "7. Explain the solution step by step.\n"
            "8. Mention time complexity when applicable.\n"
            "9. Mention space complexity when applicable.\n\n"

            "FORMAT\n\n"

            "Code:\n"
            "```python\n"
            "your solution here\n"
            "```\n\n"

            "Explanation:\n"
            "Explain the solution clearly."
        )

        return prompt

    def _build_direct_prompt(
        self,
        query: str,
    ) -> str:
        """
        Build a direct LLM prompt when retrieved documents
        are not relevant.

        Args:
            query:
                User's programming request.

        Returns:
            Direct generation prompt.
        """

        prompt = (
            "You are an expert Python programming assistant.\n\n"

            "Solve the following programming request directly.\n"
            "No relevant retrieved examples are available, "
            "so rely on your programming knowledge.\n\n"

            "USER REQUEST\n"
            + "=" * 80
            + "\n\n"
            + query
            + "\n\n"
            + "=" * 80
            + "\n\n"

            "INSTRUCTIONS\n\n"

            "1. Understand the requirements carefully.\n"
            "2. Generate a complete Python solution.\n"
            "3. Keep the implementation clean and readable.\n"
            "4. Handle reasonable edge cases.\n"
            "5. Do not invent requirements.\n"
            "6. Explain the solution step by step.\n"
            "7. Mention time complexity when applicable.\n"
            "8. Mention space complexity when applicable.\n\n"

            "FORMAT\n\n"

            "Code:\n"
            "```python\n"
            "your solution here\n"
            "```\n\n"

            "Explanation:\n"
            "Explain the solution clearly."
        )

        return prompt

    def _generate_response(
        self,
        prompt: str,
    ) -> str:
        """
        Send the generation prompt to the LLM.

        Args:
            prompt:
                Prompt to send to the LLM.

        Returns:
            LLM response.
        """

        logger.info(
            "Sending generation prompt to LLM."
        )

        response = self.llm.generate(
            prompt
        )

        logger.info(
            "Generation response received successfully."
        )

        return response

    def run(
        self,
        query: str,
    ) -> dict:
        """
        Run the complete generation workflow.

        Args:
            query:
                User's programming request.

        Returns:
            Dictionary containing:

            - query
            - response
            - retrieved_documents
            - retrieval_relevant
            - generation_mode
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        try:

            logger.info(
                "Starting Generation Workflow."
            )

            # ==================================================
            # STEP 1: RETRIEVAL
            # ==================================================

            documents = self.retriever.retrieve(
                query
            )

            logger.info(
                "Retrieved %d documents.",
                len(documents),
            )

            # ==================================================
            # STEP 2: RETRIEVAL EVALUATION
            # ==================================================

            is_relevant = self.evaluator.evaluate(
                query=query,
                documents=documents,
            )

            logger.info(
                "Retrieval relevance: %s",
                is_relevant,
            )

            # ==================================================
            # STEP 3: SELECT GENERATION PATH
            # ==================================================

            if is_relevant:

                logger.info(
                    "Relevant context found. "
                    "Using RAG generation."
                )

                prompt = self._build_rag_prompt(
                    query=query,
                    documents=documents,
                )

                generation_mode = "rag"

            else:

                logger.info(
                    "Retrieved context is not relevant. "
                    "Using direct LLM generation."
                )

                prompt = self._build_direct_prompt(
                    query=query,
                )

                generation_mode = "direct"

            # ==================================================
            # STEP 4: GENERATION
            # ==================================================

            response = self._generate_response(
                prompt
            )

            # ==================================================
            # STEP 5: RETURN RESULT
            # ==================================================

            result = {
                "query": query,
                "response": response,
                "retrieved_documents": documents,
                "retrieval_relevant": is_relevant,
                "generation_mode": generation_mode,
            }

            logger.info(
                "Generation Workflow completed successfully."
            )

            return result

        except Exception:

            logger.exception(
                "Generation Workflow failed."
            )

            raise