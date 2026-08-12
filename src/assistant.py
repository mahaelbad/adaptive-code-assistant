"""
Adaptive Code Assistant

Main orchestration layer for the Adaptive Code Assistant.

Flow:

User Request
    ↓
LLM Router
    ↓
┌───────────────────────┬───────────────────────┐
│                       │
code_explanation        code_generation
│                       │
↓                       ↓
ExplanationWorkflow     GenerationWorkflow
│                       │
↓                       ↓
Explanation             RAG / Direct Generation
└───────────────────────┴───────────────────────┘
"""

from src.router.router import LLMRouter
from src.workflows.explanation_workflow import ExplanationWorkflow
from src.workflows.generation_workflow import GenerationWorkflow
from src.utils.logger import logger


class AdaptiveCodeAssistant:
    """
    Main orchestrator for the Adaptive Code Assistant.

    The assistant:

    1. Classifies the user's request.
    2. Selects the appropriate workflow.
    3. Executes the selected workflow.
    4. Returns the final response.
    """

    def __init__(self) -> None:
        """
        Initialize the assistant and its workflows.
        """

        logger.info(
            "Initializing Adaptive Code Assistant."
        )

        # ==========================================
        # Router
        # ==========================================

        self.router = LLMRouter()

        # ==========================================
        # Code Explanation Workflow
        # ==========================================

        self.explanation_workflow = (
            ExplanationWorkflow()
        )

        # ==========================================
        # Code Generation Workflow
        # ==========================================

        self.generation_workflow = (
            GenerationWorkflow()
        )

        logger.info(
            "Adaptive Code Assistant initialized successfully."
        )

    def run(
        self,
        query: str,
        code: str = "",
    ) -> dict:
        """
        Process a user request.

        Args:
            query:
                User's request.

            code:
                User-provided code.
                Required for code explanation requests.

        Returns:
            Dictionary containing the route and response.

            For code explanation:

            {
                "route": "code_explanation",
                "response": "..."
            }

            For code generation:

            {
                "route": "code_generation",
                "response": "...",
                "generation_mode": "rag" or "direct",
                "retrieval_relevant": True or False,
                "retrieved_documents": number
            }

        Raises:
            ValueError:
                If the query is empty or code is missing
                for an explanation request.
        """

        # ==========================================
        # Validate query
        # ==========================================

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # Code is optional for generation requests.
        # Normalize it here so downstream workflows
        # receive a clean string.

        code = code.strip() if code else ""

        logger.info(
            "Starting Adaptive Code Assistant."
        )

        # ==========================================
        # STEP 1: CLASSIFY REQUEST
        # ==========================================

        logger.info(
            "Classifying user request."
        )

        route = self.router.classify(
            query
        )

        logger.info(
            "Selected workflow: %s",
            route,
        )

        # ==========================================
        # STEP 2: CODE EXPLANATION
        # ==========================================

        if route == "code_explanation":

            logger.info(
                "Running Explanation Workflow."
            )

            # Code is mandatory for explanation.
            if not code:
                raise ValueError(
                    "Code is required for "
                    "code explanation requests."
                )

            response = (
                self.explanation_workflow.run(
                    query=query,
                    code=code,
                )
            )

            result = {
                "route": route,
                "response": response,
            }

        # ==========================================
        # STEP 3: CODE GENERATION
        # ==========================================

        elif route == "code_generation":

            logger.info(
                "Running Generation Workflow."
            )

            generation_result = (
                self.generation_workflow.run(
                    query=query,
                )
            )

            result = {
                "route": route,

                "response": (
                    generation_result["response"]
                ),

                "generation_mode": (
                    generation_result[
                        "generation_mode"
                    ]
                ),

                "retrieval_relevant": (
                    generation_result[
                        "retrieval_relevant"
                    ]
                ),

                "retrieved_documents": len(
                    generation_result[
                        "retrieved_documents"
                    ]
                ),
            }

        # ==========================================
        # STEP 4: INVALID ROUTE
        # ==========================================

        else:

            logger.error(
                "Unsupported route returned by router: %s",
                route,
            )

            raise ValueError(
                f"Unsupported route: {route}"
            )

        # ==========================================
        # STEP 5: COMPLETE
        # ==========================================

        logger.info(
            "Adaptive Code Assistant completed successfully."
        )

        return result