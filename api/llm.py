from fastapi import APIRouter

from core.llm_client import llm_client
from core.models import LLMRequestConfig

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/test")
async def test_llm_config(config: LLMRequestConfig):
    """Validate a user-provided OpenAI-compatible LLM config.

    The API key is only used for this request and is never persisted.
    """
    return llm_client.test_request_config(config)


@router.get("/status")
async def llm_status():
    """Return built-in LLM config status without exposing sensitive data."""
    has_key = bool(llm_client.openai_api_key)
    has_url = bool(llm_client.openai_base_url)
    has_model = bool(llm_client.model)
    is_available = has_key and has_url and has_model

    # Mask API key: show only first 4 and last 4 chars
    masked_key = ""
    if llm_client.openai_api_key:
        key = llm_client.openai_api_key
        if len(key) > 8:
            masked_key = key[:4] + "*" * (len(key) - 8) + key[-4:]
        else:
            masked_key = "****"

    return {
        "available": is_available,
        "provider": llm_client.provider,
        "provider_label": llm_client.provider_label,
        "model": llm_client.model or "",
        "base_url": llm_client.openai_base_url or "",
        "api_key_masked": masked_key,
        "has_key": has_key,
        "has_url": has_url,
        "has_model": has_model,
    }
