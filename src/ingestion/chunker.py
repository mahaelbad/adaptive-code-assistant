"""
Document Chunker

This module is responsible for splitting LangChain Documents
into smaller chunks suitable for embedding and retrieval.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.utils.logger import logger


class DocumentChunker:
    """
    Split LangChain Documents into smaller chunks.
    """

    def __init__(self) -> None:
        """
        Initialize the Recursive Character Text Splitter.
        """

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        logger.info(
            "DocumentChunker initialized "
            "(chunk_size=%d, overlap=%d).",
            settings.chunk_size,
            settings.chunk_overlap
        )

    def split_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:
        """
        Split LangChain Documents into chunks.

        Args:
            documents: List of LangChain Documents.

        Returns:
            List[Document]: Chunked documents.
        """

        try:
            chunks = self.splitter.split_documents(documents)

            logger.info(
                "Successfully split %d documents into %d chunks.",
                len(documents),
                len(chunks)
            )

            return chunks

        except Exception:
            logger.exception("Failed to split documents.")
            raise

    def chunk_statistics(
        self,
        chunks: list[Document]
    ) -> None:
        """
        Display statistics about generated chunks.

        Args:
            chunks: List of chunked documents.
        """

        logger.info("========== Chunk Statistics ==========")
        logger.info("Total Chunks: %d", len(chunks))

        if not chunks:
            logger.warning("No chunks available.")
            return

        first_chunk = chunks[0]

        logger.info(
            "First Chunk Length: %d characters",
            len(first_chunk.page_content)
        )

        logger.info(
            "First Chunk Metadata: %s",
            first_chunk.metadata
        )

        print("\n" + "=" * 80)
        print("FIRST CHUNK PREVIEW")
        print("=" * 80)
        print(first_chunk.page_content[:700])
        print("=" * 80)