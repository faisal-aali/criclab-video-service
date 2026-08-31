from functools import lru_cache
from os import environ
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_storage_dir() -> Path:
    """Runtime uploads/artifacts live outside the git repo (OS user data dir)."""
    return Path.home() / ".local" / "share" / "criclab"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "criclab"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_embed_model: str = "nomic-embed-text"
    # `local` → Ollama. `production` → Bedrock (chat + embeddings).
    app_env: str = "local"
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "qwen.qwen3-coder-30b-a3b-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    # Leave empty to use ~/.local/share/criclab (outside the repo).
    storage_dir: str = ""
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    default_meters_per_pixel: float | None = None

    # --- Auth ---
    # No default for the signing key: an app that silently boots on a shared
    # fallback secret issues forgeable tokens. Startup fails loudly instead.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 30

    # --- Email (OTP, password reset, notifications) ---
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    email_from: str = ""
    email_from_name: str = "CricLab"
    # Where links in emails point — the frontend, not this API.
    app_base_url: str = "http://localhost:5173"

    # Cloudinary (processed video + PDF hosting)
    cloudinary_url: str | None = None
    cloudinary_cloud_name: str | None = None
    cloudinary_api_key: str | None = None
    cloudinary_api_secret: str | None = None

    @field_validator(
        "jwt_secret",
        "smtp_host",
        "smtp_user",
        "smtp_pass",
        "email_from",
        mode="before",
    )
    @classmethod
    def empty_keep_str(cls, v: Any) -> str:
        # These fields are `str`, not Optional. Empty env on Vercel arrives as
        # None / "" and must not be coerced to None or Settings fails to boot.
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        return str(v)

    @field_validator(
        "default_meters_per_pixel",
        "cloudinary_url",
        "cloudinary_cloud_name",
        "cloudinary_api_key",
        "cloudinary_api_secret",
        "aws_access_key_id",
        "aws_secret_access_key",
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

    @field_validator("app_env", mode="before")
    @classmethod
    def _app_env(cls, v: Any) -> str:
        # Vercel injects VERCEL=1. If APP_ENV is not set there, use production
        # (Bedrock) rather than the local Ollama default.
        if environ.get("VERCEL") and not (environ.get("APP_ENV") or "").strip():
            return "production"
        value = str(v or "local").strip().lower()
        if value in ("prod", "production"):
            return "production"
        if value in ("local", "dev", "development"):
            return "local"
        raise ValueError("APP_ENV must be 'local' or 'production'")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def llm_provider(self) -> str:
        return "bedrock" if self.is_production else "ollama"

    @property
    def embed_model_id(self) -> str:
        """Which embedding model the assistant index is keyed on."""
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
        elif environ.get("VERCEL"):
            path = Path("/tmp/criclab")
        else:
            path = default_storage_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def email_configured(self) -> bool:
        """Email is optional in development; the app degrades to logging OTPs."""
        return bool(self.smtp_host and self.smtp_user and self.smtp_pass and self.email_from)

    @property
    def auth_configured(self) -> bool:
        return bool(self.jwt_secret)

    @property
    def cloudinary_configured(self) -> bool:
        if self.cloudinary_url:
            return True
        return bool(self.cloudinary_cloud_name and self.cloudinary_api_key and self.cloudinary_api_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
