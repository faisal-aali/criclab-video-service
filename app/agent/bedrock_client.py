"""Amazon Bedrock runtime — production chat + embeddings.

Credentials come from Settings (`.env`) or the normal AWS chain. Never send
keys to the browser; this module is server-side only.
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any, Iterator

from app.config import get_settings

_SENTINEL = object()


def _client_kwargs() -> dict[str, str]:
    settings = get_settings()
    kwargs: dict[str, str] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return kwargs


@lru_cache
def _runtime():
    import boto3

    return boto3.client("bedrock-runtime", **_client_kwargs())


def _sts():
    import boto3

    return boto3.client("sts", **_client_kwargs())


def _converse_kwargs(
    prompt: str,
    system: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    settings = get_settings()
    body: dict[str, Any] = {
        "modelId": settings.bedrock_model_id,
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        "inferenceConfig": {
            "maxTokens": max(1, int(max_tokens)),
            "temperature": 0.3,
        },
    }
    if system:
        body["system"] = [{"text": system}]
    return body


def _extract_text(response: dict[str, Any]) -> str:
    blocks = ((response.get("output") or {}).get("message") or {}).get("content") or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("text"):
            parts.append(block["text"])
    return "".join(parts).strip()


def _converse_sync(prompt: str, system: str | None, max_tokens: int) -> str:
    response = _runtime().converse(**_converse_kwargs(prompt, system, max_tokens))
    return _extract_text(response)


def _stream_sync(prompt: str, system: str | None, max_tokens: int) -> Iterator[str]:
    response = _runtime().converse_stream(**_converse_kwargs(prompt, system, max_tokens))
    for event in response.get("stream") or []:
        delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
        piece = delta.get("text")
        if piece:
            yield piece


async def converse(
    prompt: str,
    system: str | None = None,
    *,
    max_tokens: int = 500,
    timeout_s: float = 300.0,
) -> str:
    return await asyncio.wait_for(
        asyncio.to_thread(_converse_sync, prompt, system, max_tokens),
        timeout=timeout_s,
    )


async def converse_stream(
    prompt: str,
    system: str | None = None,
    *,
    max_tokens: int = 500,
    timeout_s: float = 60.0,
):
    iterator = _stream_sync(prompt, system, max_tokens)

    async def _next() -> object:
        return await asyncio.to_thread(next, iterator, _SENTINEL)

    while True:
        piece = await asyncio.wait_for(_next(), timeout=timeout_s)
        if piece is _SENTINEL:
            break
        yield str(piece)


async def bedrock_available() -> dict[str, Any]:
    """Credentials + region check. Does not invoke a model (that costs money)."""
    settings = get_settings()
    out: dict[str, Any] = {
        "ok": False,
        "provider": "bedrock",
        "app_env": settings.app_env,
        "region": settings.aws_region,
        "model_id": settings.bedrock_model_id,
        "embedding_model_id": settings.bedrock_embedding_model_id,
    }
    try:
        identity = await asyncio.to_thread(_sts().get_caller_identity)
        out["ok"] = True
        out["account"] = identity.get("Account")
        out["arn"] = identity.get("Arn")
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _embed_one_sync(text: str) -> list[float]:
    settings = get_settings()
    response = _runtime().invoke_model(
        modelId=settings.bedrock_embedding_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text, "normalize": True}),
    )
    payload = json.loads(response["body"].read())
    vector = payload.get("embedding")
    if not isinstance(vector, list):
        raise RuntimeError("Bedrock embedding response had no vector")
    return [float(x) for x in vector]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(lambda: [_embed_one_sync(t) for t in texts])
