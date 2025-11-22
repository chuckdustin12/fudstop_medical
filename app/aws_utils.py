"""Optional AWS utilities for OCR/text extraction.

These helpers are intentionally lightweight so the service can run locally
without AWS credentials. When ``TEXTRACT_ENABLED=true`` in the environment,
we will attempt to extract text from image-only PDFs using Amazon Textract.
"""

import logging
from functools import lru_cache
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger("medical_audit.aws")


@lru_cache(maxsize=1)
def get_textract_client(region: str):
    """Return a cached Textract client for the given region."""
    return boto3.client("textract", region_name=region)


def textract_document_text(file_bytes: bytes, *, region: str) -> str:
    """
    Extract text using Amazon Textract. Safe to call even if credentials are
    missing; errors are logged and an empty string is returned so the caller
    can continue gracefully.
    """
    try:
        client = get_textract_client(region)
        response = client.detect_document_text(Document={"Bytes": file_bytes})
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - network
        logger.error("Textract request failed: %s", exc)
        return ""

    lines = [
        block["Text"]
        for block in response.get("Blocks", [])
        if block.get("BlockType") == "LINE" and block.get("Text")
    ]
    return "\n".join(lines).strip()

