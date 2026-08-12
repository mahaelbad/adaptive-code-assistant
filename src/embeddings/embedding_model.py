"""
Embedding Model Loader

This module is responsible for loading and providing
a singleton embedding model instance.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings
from src.utils.logger import logger


class EmbeddingModel:
    """
    Singleton class responsible for loading and providing
    the embedding model.
    """

    _model: HuggingFaceEmbeddings | None = None

    def __init__(self) -> None:
        """
        Initialize the embedding model only once.
        """

        if EmbeddingModel._model is None:
            try:
                logger.info(
                    "Loading embedding model: %s",
                    settings.embedding_model
                )

                EmbeddingModel._model = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model,
                    model_kwargs={
                        "device": "cpu"
                    },
                    encode_kwargs={
                        "normalize_embeddings": True
                    }
                )

                logger.info("Embedding model loaded successfully.")

            except Exception:
                logger.exception("Failed to load embedding model.")
                raise

        else:
            logger.info("Using cached embedding model.")

    def get_model(self) -> HuggingFaceEmbeddings:
        """
        Return the singleton embedding model.

        Returns:
            HuggingFaceEmbeddings
        """

        return EmbeddingModel._model