"""
Knowledge Updater

This module is responsible for updating the assistant's
knowledge base by:

1. Loading source documents.
2. Splitting documents into chunks.
3. Building a FAISS vector store.
4. Saving the updated vector store.

Flow:

Documents
    ↓
Document Chunker
    ↓
Chunks
    ↓
FAISS Vector Store
    ↓
Save
"""

from langchain_core.documents import Document

from src.ingestion.chunker import DocumentChunker
from src.vectorstore.vector_store import VectorStoreManager
from src.utils.logger import logger


class KnowledgeUpdater:
    """
    Manage the process of updating the knowledge base.
    """

    def __init__(self) -> None:
        """
        Initialize the knowledge updater.
        """

        self.chunker = DocumentChunker()

        self.vector_store = VectorStoreManager()

        logger.info(
            "KnowledgeUpdater initialized successfully."
        )

    def update(
        self,
        documents: list[Document],
    ) -> None:
        """
        Update the knowledge base from source documents.

        Args:
            documents:
                List of LangChain Documents.

        Raises:
            ValueError:
                If no documents are provided.
        """

        if not documents:
            raise ValueError(
                "No documents provided for knowledge update."
            )

        try:

            logger.info(
                "Starting knowledge base update."
            )

            # ==========================================
            # STEP 1: CHUNK DOCUMENTS
            # ==========================================

            logger.info(
                "Splitting %d source documents into chunks.",
                len(documents),
            )

            chunks = self.chunker.split_documents(
                documents
            )

            logger.info(
                "Generated %d document chunks.",
                len(chunks),
            )

            if not chunks:
                raise ValueError(
                    "No chunks were generated from the documents."
                )

            # ==========================================
            # STEP 2: SHOW CHUNK STATISTICS
            # ==========================================

            self.chunker.chunk_statistics(
                chunks
            )

            # ==========================================
            # STEP 3: BUILD FAISS VECTOR STORE
            # ==========================================

            logger.info(
                "Building FAISS vector store from chunks."
            )

            self.vector_store.build(
                chunks
            )

            # ==========================================
            # STEP 4: SAVE VECTOR STORE
            # ==========================================

            logger.info(
                "Saving FAISS vector store."
            )

            self.vector_store.save()

            # ==========================================
            # STEP 5: COMPLETED
            # ==========================================

            logger.info(
                "Knowledge base updated successfully."
            )

        except Exception:

            logger.exception(
                "Knowledge base update failed."
            )

            raise