from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "ResearchCopilot"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_papers: str = "research_papers"
    chroma_collection_memory: str = "adaptive_memory"

    # DynamoDB
    aws_region: str = "us-east-1"
    aws_access_key_id: str = "dummy"
    aws_secret_access_key: str = "dummy"
    dynamodb_endpoint_url: str = "http://localhost:8002"
    dynamodb_papers_table: str = "ResearchPapers"
    dynamodb_sessions_table: str = "UserSessions"

    # Ollama / LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:35b"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Semantic Scholar
    semantic_scholar_api_key: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "researchcopilot"


@lru_cache
def get_settings() -> Settings:
    return Settings()
