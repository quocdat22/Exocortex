"""Centralized configuration for Exocortex RAG system.

All configurable parameters are defined here. Secrets are loaded from .env file.
Technical parameters have sensible defaults that can be overridden via environment
variables.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (DeepSeek) ---
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- Embedding (Ollama) ---
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dim: int = 1024
    embedding_num_ctx: int = 32768  # qwen3-embedding:0.6b supports 32k context

    # --- Chunking ---
    chunk_size: int = 512  # tokens per chunk
    chunk_overlap: int = 50  # overlap tokens between chunks

    # --- Retrieval ---
    top_k: int = 5  # number of chunks to retrieve

    # --- ChromaDB ---
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "exocortex_ebooks"

    # --- Paths ---
    ebooks_dir: str = "./data/ebooks"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def ebooks_path(self) -> Path:
        """Return ebooks directory as a Path object."""
        return Path(self.ebooks_dir)

    @property
    def chroma_path(self) -> Path:
        """Return ChromaDB directory as a Path object."""
        return Path(self.chroma_persist_dir)


def get_settings() -> Settings:
    """Factory function to create Settings instance.

    Use this instead of instantiating Settings directly to allow for
    dependency injection in tests.
    """
    return Settings()
