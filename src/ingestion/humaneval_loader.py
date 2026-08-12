import json
from pathlib import Path

from datasets import load_dataset
from langchain_core.documents import Document

from config.settings import settings
from src.ingestion.base_loader import BaseLoader
from src.utils.logger import logger




class HumanEvalLoader(BaseLoader):
    """
    HumanEval Dataset Loader.

    Responsibilities:
    -----------------
    1. Download the HumanEval dataset (if not cached).
    2. Load the dataset from the local cache.
    3. Convert dataset samples into LangChain Documents.
    """

    def __init__(self):
        self.dataset_name = settings.dataset_name
        self.raw_data_path = Path("data/raw/humaneval.json")

    def load_dataset(self) -> list[dict]:
        """
        Load the HumanEval dataset.

        If the dataset exists locally, load it.
        Otherwise, download it from Hugging Face,
        save it locally, then return it.
        """

        try:
            # Load from local cache
            if self.raw_data_path.exists():

                with open(
                    self.raw_data_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    dataset = json.load(file)

                logger.info("Loaded dataset from local cache.")

                return dataset

            logger.info("Downloading HumanEval dataset...")

            dataset = load_dataset(self.dataset_name)["test"]
            dataset = list(dataset)

            # Create directory if it doesn't exist
            self.raw_data_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Save dataset locally
            with open(
                self.raw_data_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    dataset,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            logger.info("Dataset downloaded and saved successfully.")

            return dataset

        except Exception:
            logger.exception("Failed to load HumanEval dataset.")
            raise

    def convert_to_documents(
        self,
        dataset: list[dict]
    ) -> list[Document]:
        """
        Convert dataset samples into LangChain Documents.
        """

        documents: list[Document] = []

        try:
            for sample in dataset:

                content = f"""
Task ID:
{sample['task_id']}

Problem:
{sample['prompt']}

Function Name:
{sample['entry_point']}

Reference Solution:
{sample['canonical_solution']}
""".strip()

                metadata = {
                    "task_id": sample["task_id"],
                    "entry_point": sample["entry_point"],
                    "source": "HumanEval"
                }

                documents.append(
                    Document(
                        page_content=content,
                        metadata=metadata
                    )
                )

            logger.info(
                "Converted %d documents successfully.",
                len(documents)
            )

            return documents

        except Exception:
            logger.exception("Failed to convert dataset to documents.")
            raise

    def load(self) -> list[Document]:
        """
        Load the dataset and convert it into LangChain Documents.
        """

        dataset = self.load_dataset()

        documents = self.convert_to_documents(dataset)

        return documents