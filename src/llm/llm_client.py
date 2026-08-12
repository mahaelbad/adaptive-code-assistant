"""
LLM Client

This module provides a single interface for communicating
with the Large Language Model through OpenRouter.
"""

import time

from langchain_openrouter import ChatOpenRouter

from config.settings import settings
from src.utils.logger import logger


class LLMClient:
    """
    Client responsible for interacting with the LLM.
    """

    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_attempts: int = 3,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            temperature:
                Controls randomness of the model output.

            max_tokens:
                Maximum number of tokens generated.

            max_attempts:
                Maximum number of attempts when generating
                a response.
        """

        try:
            self.max_attempts = max_attempts

            self.model = ChatOpenRouter(
                model=settings.llm_model,
                api_key=settings.openrouter_api_key,
                temperature=temperature,
                max_tokens=max_tokens,

                # LangChain internal retries.
                max_retries=2,
            )

            logger.info(
                "LLM initialized successfully: %s",
                settings.llm_model,
            )

        except Exception:
            logger.exception(
                "Failed to initialize LLM."
            )
            raise

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt:
                Prompt sent to the model.

        Returns:
            Generated response as text.

        Raises:
            Exception:
                If all generation attempts fail.
        """

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        last_exception = None

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):

            try:
                logger.info(
                    "Sending request to LLM "
                    "(attempt %d/%d).",
                    attempt,
                    self.max_attempts,
                )

                response = self.model.invoke(
                    prompt
                )

                content = response.content

                # Handle possible empty responses.
                if content is None:
                    content = ""

                content = str(content).strip()

                if not content:
                    logger.warning(
                        "LLM returned an empty response "
                        "(attempt %d/%d).",
                        attempt,
                        self.max_attempts,
                    )

                    if attempt < self.max_attempts:
                        wait_time = 2 ** attempt

                        logger.info(
                            "Retrying after %d seconds...",
                            wait_time,
                        )

                        time.sleep(wait_time)

                        continue

                    raise ValueError(
                        "LLM returned an empty response."
                    )

                logger.info(
                    "LLM response generated successfully."
                )

                return content

            except Exception as exc:

                last_exception = exc

                error_message = str(exc).lower()

                # Detect rate-limit/provider errors.
                is_rate_limit = (
                    "too many requests" in error_message
                    or "429" in error_message
                    or "rate limit" in error_message
                )

                if is_rate_limit:
                    logger.warning(
                        "LLM rate limit/provider limit "
                        "encountered "
                        "(attempt %d/%d).",
                        attempt,
                        self.max_attempts,
                    )

                else:
                    logger.warning(
                        "LLM generation failed "
                        "(attempt %d/%d): %s",
                        attempt,
                        self.max_attempts,
                        exc,
                    )

                # Stop if this was the last attempt.
                if attempt >= self.max_attempts:
                    break

                # Exponential backoff.
                wait_time = 2 ** attempt

                logger.info(
                    "Waiting %d seconds before retry...",
                    wait_time,
                )

                time.sleep(wait_time)

        logger.error(
            "LLM generation failed after %d attempts.",
            self.max_attempts,
        )

        if last_exception is not None:
            raise last_exception

        raise RuntimeError(
            "LLM generation failed."
        )