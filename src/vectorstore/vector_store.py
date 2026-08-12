"""
FAISS Vector Store Manager

This module is responsible for:
1. Building the FAISS vector store.
2. Saving the vector store.
3. Loading the vector store.
4. Performing similarity search.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from src.embeddings.embedding_model import EmbeddingModel
from src.utils.logger import logger


class VectorStoreManager:
    """
    Manage the FAISS Vector Store.
    """

    def __init__(self) -> None:
        """
        Initialize the vector store manager.
        """

        self.embedding_model = EmbeddingModel().get_model()

        self.vector_store = None

        self.vector_store_path = Path(settings.vector_store_path)

        self.index_name = settings.index_name

        logger.info("VectorStoreManager initialized.")

    def build(
        self,
        documents: list[Document]
    ) -> FAISS:
        """
        Build a FAISS index from documents.

        Args:
            documents:
                List of LangChain Documents.

        Returns:
            FAISS vector store.
        """

        try:

            logger.info(
                "Building FAISS index from %d documents...",
                len(documents)
            )

            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )

            logger.info("FAISS index created successfully.")

            return self.vector_store

        except Exception:

            logger.exception("Failed to build FAISS index.")

            raise

    def save(self) -> None:
        """
        Save the FAISS index to disk.
        """

        try:

            if self.vector_store is None:
                raise ValueError(
                    "Vector store has not been built yet."
                )

            self.vector_store_path.mkdir(
                parents=True,
                exist_ok=True
            )

            self.vector_store.save_local(
                folder_path=str(self.vector_store_path),
                index_name=self.index_name
            )

            logger.info(
                "FAISS index saved successfully."
            )

        except Exception:

            logger.exception(
                "Failed to save FAISS index."
            )

            raise

    def load(self) -> FAISS:
        """
        Load the FAISS index from disk.

        Returns:
            FAISS vector store.
        """

        try:

            self.vector_store = FAISS.load_local(
                folder_path=str(self.vector_store_path),
                embeddings=self.embedding_model,
                index_name=self.index_name,
                allow_dangerous_deserialization=True
            )

            logger.info(
                "FAISS index loaded successfully."
            )

            return self.vector_store

        except Exception:

            logger.exception(
                "Failed to load FAISS index."
            )

            raise

    def similarity_search(
        self,
        query: str,
        k: int = 3
    ) -> list[Document]:
        """
        Search for the most similar documents.

        Args:
            query:
                User query.

            k:
                Number of retrieved documents.

        Returns:
            List of Documents.
        """

        try:

            if self.vector_store is None:
                raise ValueError(
                    "Vector store is not loaded."
                )

            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )

            logger.info(
                "Retrieved %d similar documents.",
                len(results)
            )

            return results

        except Exception:

            logger.exception(
                "Similarity search failed."
            )

            raise