from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env to os.environ so OpenAI, Tavily and LangSmith clients find their keys
load_dotenv()


class Settings(BaseSettings):
    """All runtime configuration comes from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NETPLAN_", extra="ignore")

    # Any provider supported by init_chat_model, e.g.
    #   openai:gpt-5-mini
    #   bedrock_converse:us.anthropic.claude-sonnet-4-5-20250929-v1:0
    coordinator_model: str = "openai:gpt-4.1-mini"
    specialist_model: str = "openai:gpt-5-nano"
    fallback_model: str | None = None

    embeddings_provider: str = "openai"  # openai or bedrock
    embeddings_model: str = "text-embedding-3-small"

    docs_dir: Path = Path("data/specs")
    index_dir: Path = Path("data/index")
    db_path: Path = Path("data/telecom_params.db")

    max_web_searches: int = 5
    max_model_calls: int = 25
    retrieval_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()


def build_embeddings(s: Settings):
    if s.embeddings_provider == "bedrock":
        from langchain_aws import BedrockEmbeddings  # optional dependency

        return BedrockEmbeddings(model_id=s.embeddings_model)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=s.embeddings_model)
