"""Tests for Phase 1: Configuration module."""

from exocortex.config import Settings, get_settings


def test_settings_default_values():
    """Settings should load with sensible defaults."""
    settings = Settings(deepseek_api_key="test-key")

    assert settings.deepseek_api_key == "test-key"
    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.embedding_model == "qwen3-embedding:0.6b"
    assert settings.embedding_dim == 1024
    assert settings.chunking_strategy == "recursive"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 50
    assert settings.top_k == 5
    assert settings.llm_temperature == 0.1
    assert settings.chroma_collection_name == "exocortex_ebooks"


def test_settings_env_override(monkeypatch):
    """Settings should be overridable via environment variables."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "override-key")
    monkeypatch.setenv("CHUNKING_STRATEGY", "fixed")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("TOP_K", "10")

    settings = Settings()

    assert settings.deepseek_api_key == "override-key"
    assert settings.chunking_strategy == "fixed"
    assert settings.chunk_size == 256
    assert settings.top_k == 10


def test_settings_path_properties():
    """Path properties should return Path objects."""
    settings = Settings(deepseek_api_key="test-key")

    assert settings.ebooks_path.name == "ebooks"
    assert settings.chroma_path.name == "chroma_data"


def test_get_settings_returns_settings():
    """get_settings() should return a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_settings_session_defaults():
    """Settings should include default session database path and history window."""
    settings = Settings(deepseek_api_key="test-key")
    assert settings.sessions_db_path == "./data/sessions.db"
    assert settings.chat_history_window == 3
    assert settings.sessions_path.name == "sessions.db"


def test_settings_session_env_override(monkeypatch):
    """Settings should allow overriding session settings via environment variables."""
    monkeypatch.setenv("SESSIONS_DB_PATH", "./custom/sessions.db")
    monkeypatch.setenv("CHAT_HISTORY_WINDOW", "5")

    settings = Settings(deepseek_api_key="test-key")
    assert settings.sessions_db_path == "./custom/sessions.db"
    assert settings.chat_history_window == 5

