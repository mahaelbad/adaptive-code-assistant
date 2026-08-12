"""
Retriever Module

This module provides semantic retrieval over the FAISS vector store.
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.utils.logger import logger
from src.vectorstore.vector_store import VectorStoreManager


class Retriever:
    """
    Semantic Retriever built on top of FAISS.
    """

    def __init__(
        self,
        k: int = 3
    ) -> None:
        """
        Initialize the retriever.

        Args:
            k: Number of retrieved documents.
        """

        self.vector_store_manager = VectorStoreManager()

        self.vector_store = self.vector_store_manager.load()

        self.k = k

        self.retriever: BaseRetriever = (
            self.vector_store.as_retriever(
                search_kwargs={
                    "k": self.k
                }
            )
        )

        logger.info(
            "Retriever initialized (k=%d).",
            self.k
        )

    def retrieve(
        self,
        query: str
    ) -> list[Document]:
        """
        Retrieve relevant documents.

        Args:
            query:
                User query.

        Returns:
            List of relevant documents.
        """

        try:

            documents = self.retriever.invoke(query)

            logger.info(
                "Retrieved %d documents.",
                len(documents)
            )

            return documents

        except Exception:

            logger.exception(
                "Retrieval failed."
            )

            raise