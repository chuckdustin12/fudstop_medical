"""Application settings for medical audit service."""

from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    app_name: str = Field("Medical Audit Service", description="Service title")
    aws_region: str = Field("us-east-1", description="AWS region for Textract")
    textract_enabled: bool = Field(
        False,
        description=(
            "If true, attempt OCR with Amazon Textract when PDFs have no embedded text."
        ),
    )
    max_pdf_pages: Optional[int] = Field(
        20,
        description="Limit number of pages processed locally to keep latency low in serverless runtimes.",
    )

    class Config:
        env_prefix = "MED_AUDIT_"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

