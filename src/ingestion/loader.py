from src.ingestion.humaneval_loader import HumanEvalLoader


class DataLoader:
    """
    Facade for loading project documents.
    """

    def __init__(self):
        self.loader = HumanEvalLoader()

    def load(self):
        return self.loader.load()