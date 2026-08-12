"""
Application Settings

This module loads all project configurations from the .env file
using Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application configuration.
    """

    # ==========================
    # Dataset
    # ==========================
    dataset_name: str

    # ==========================
    # Embedding Model
    # ==========================
    embedding_model: str

    # ==========================
    # Vector Store
    # ==========================
    vector_store_path: str
    index_name: str

    # ==========================
    # Prompts
    # ==========================
    prompts_dir: str
    system_prompt_file: str

    # ==========================
    # Text Chunking
    # ==========================
    chunk_size: int
    chunk_overlap: int

    # ==========================
    # Retrieval
    # ==========================
    top_k: int

    # ==========================
    # Vector Database
    # ==========================
    vector_db_path: str

    # ==========================
    # LLM
    # ==========================
    openrouter_api_key: str
    llm_model: str = "openrouter/free"
    temperature: float = 0.0

    # ==========================
    # Application
    # ==========================
    app_name: str = "Adaptive Code Assistant"
    debug: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()