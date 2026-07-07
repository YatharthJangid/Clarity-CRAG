from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.config import settings

primary_llm = ChatGoogleGenerativeAI(
    model=settings.chat_model,
    temperature=0,
    google_api_key=settings.gemini_api_key,
    # Gemini's free tier is 5 requests/minute; without capping retries, the client's
    # own exponential backoff burns tens of seconds per call on a 429 before
    # `.with_fallbacks()` below ever gets a chance to hand off to Groq.
    max_retries=1,
)
fallback_llm = ChatGroq(model=settings.fallback_chat_model, temperature=0, groq_api_key=settings.groq_api_key)

# Every plain (non-structured) call in the graph goes through Gemini first and
# drops to Groq's Qwen model on error (rate limit, outage, etc).
llm = primary_llm.with_fallbacks([fallback_llm])


def structured(schema):
    """Build a Gemini-primary/Groq-fallback Runnable that returns `schema`.

    `.with_structured_output()` lives on `BaseChatModel`, not on the
    `RunnableWithFallbacks` that `llm` is - so each provider's structured chain is
    built separately here and stitched together with `.with_fallbacks()`.

    Qwen's tool-calling occasionally misfires on a structured-output call
    (`tool_use_failed`) even when the service itself is healthy, so the fallback
    gets one quick retry of its own on top of taking over for Gemini.
    """
    fallback_structured = fallback_llm.with_structured_output(schema).with_retry(stop_after_attempt=2)
    return primary_llm.with_structured_output(schema).with_fallbacks([fallback_structured])
