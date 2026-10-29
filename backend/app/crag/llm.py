from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from app.config import settings

primary_llm = ChatGoogleGenerativeAI(
    model=settings.chat_model,
    temperature=0,
    google_api_key=settings.gemini_api_key,
    # Gemini's free tier is 5 requests/minute; without capping retries, the client's
    # own exponential backoff burns tens of seconds per call on a 429 before
    # `.with_fallbacks()` below ever gets a chance to hand off to the next model.
    max_retries=1,
)
fallback_llm = ChatGroq(model=settings.fallback_chat_model, temperature=0, groq_api_key=settings.groq_api_key)


def _build_hf_chat(repo_id: str) -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        provider="auto",
        huggingfacehub_api_token=settings.hf_token,
        max_new_tokens=1024,
    )
    return ChatHuggingFace(llm=endpoint)


# ---------------------------------------------------------------------------
# Option A: Gemini is primary, Groq is fallback.
# Bypasses HuggingFace Inference Providers which require paid credits (402).
# ---------------------------------------------------------------------------
primary_llm = ChatGoogleGenerativeAI(
    model=settings.chat_model,
    temperature=0,
    google_api_key=settings.gemini_api_key,
    max_retries=1,
)
fallback_llm = ChatGroq(
    model=settings.fallback_chat_model,
    temperature=0,
    groq_api_key=settings.groq_api_key,
)

llm = primary_llm.with_fallbacks([fallback_llm])


def structured(schema):
    """Build a Runnable that returns `schema`, trying Gemini first and falling through to Groq."""
    def _coerce(res):
        if res is None:
            raise ValueError(f"Model returned None for {schema.__name__}")
        if isinstance(res, dict):
            return schema(**res)
        return res

    primary_structured = (primary_llm.with_structured_output(schema) | RunnableLambda(_coerce))
    fallback_structured = (
        (fallback_llm.with_structured_output(schema) | RunnableLambda(_coerce))
        .with_retry(stop_after_attempt=2)
    )

    return primary_structured.with_fallbacks([fallback_structured])
