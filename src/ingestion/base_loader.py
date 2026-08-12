from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    """

    @abstractmethod
    def load_dataset(self):
        """
        Load the raw dataset.
        """
        pass

    @abstractmethod
    def convert_to_documents(self, dataset) -> list[Document]:
        """
        Convert the dataset into LangChain Documents.
        """
        pass

    @abstractmethod
    def load(self) -> list[Document]:
        """
        Load and return LangChain Documents.
        """
        pass


    