"""
LLM Router

Classifies user requests into the appropriate workflow.
"""

from typing import Literal

from src.llm.llm_client import LLMClient
from src.router.prompts import ROUTER_PROMPT
from src.utils.logger import logger


RouteType = Literal[
    "code_explanation",
    "code_generation",
]


class LLMRouter:
    """
    Route user requests to the appropriate workflow.
    """

    def __init__(self) -> None:
        """
        Initialize the router.
        """

        self.llm = LLMClient(
            temperature=0.0,
            max_tokens=50,
        )

        logger.info(
            "LLM Router initialized successfully."
        )

    def classify(self, query: str) -> RouteType:
        """
        Classify a user request.

        Args:
            query:
                User's request.

        Returns:
            Either 'code_explanation' or 'code_generation'.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        prompt = ROUTER_PROMPT.format(
            query=query.strip()
        )

        response = self.llm.generate(prompt)

        logger.info(
            "Raw router response: %r",
            response,
        )

        classification = (
            response
            .strip()
            .lower()
            .replace("`", "")
            .strip()
        )

        if classification == "code_explanation":
            route: RouteType = "code_explanation"

        elif classification == "code_generation":
            route = "code_generation"

        else:
            logger.error(
                "Invalid router response: %r",
                response,
            )

            raise ValueError(
                "Router returned an invalid classification: "
                f"{response!r}"
            )

        logger.info(
            "Request classified as: %s",
            route,
        )

        return route