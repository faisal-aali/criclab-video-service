from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_storage_dir() -> Path:
    """Runtime uploads/artifacts live outside the git repo (OS user data dir)."""
    return Path.home() / ".local" / "share" / "criclab"


class Settings(BaseSettings):
    """Env for workers only. Auth, SMTP, and CORS stay on criclab-web-backend."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "criclab"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_embed_model: str = "nomic-embed-text"
    # `local` → Ollama. `production` → Bedrock (coaching notes in the pipeline).
    app_env: str = "local"
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "qwen.qwen3-coder-30b-a3b-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    ec2_instance_id: str | None = None
    ec2_region: str | None = None
    ec2_idle_stop_seconds: int = 60
    # Leave empty to use ~/.local/share/criclab (must match the website API).
    storage_dir: str = ""
    default_meters_per_pixel: float | None = None
    # Same cap as the website API. Only this process increments the Mongo counter.
    daily_video_quota: int = 60

    s3_bucket: str | None = None
    s3_region: str | None = None

    @field_validator(
        "default_meters_per_pixel",
        "s3_bucket",
        "s3_region",
        "aws_access_key_id",
        "aws_secret_access_key",
        "ec2_instance_id",
        "ec2_region",
        mode="before",
    )
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("daily_video_quota", mode="before")
    @classmethod
    def _daily_quota(cls, v: Any) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 60
        try:
            n = int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError("DAILY_VIDEO_QUOTA must be a positive integer") from exc
        if n < 1:
            raise ValueError("DAILY_VIDEO_QUOTA must be at least 1")
        return n

    @field_validator("app_env", mode="before")
    @classmethod
    def _app_env(cls, v: Any) -> str:
        value = str(v or "local").strip().lower()
        if value in ("prod", "production"):
            return "production"
        if value in ("local", "dev", "development"):
            return "local"
        raise ValueError("APP_ENV must be 'local' or 'production'")

    @property
    def ec2_stop_region(self) -> str:
        return (self.ec2_region or self.aws_region).strip()

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def llm_provider(self) -> str:
        return "bedrock" if self.is_production else "ollama"

    @property
    def embed_model_id(self) -> str:
        if self.is_production:
            return self.bedrock_embedding_model_id
        return self.ollama_embed_model

    @property
    def storage_path(self) -> Path:
        raw = (self.storage_dir or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = (Path(__file__).resolve().parent.parent / path).resolve()
        else:
            path = default_storage_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def s3_configured(self) -> bool:
        return bool(self.s3_bucket and self.s3_region)


@lru_cache
def get_settings() -> Settings:
    return Settings()
