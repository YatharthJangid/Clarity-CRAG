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


# A loop of free, high-volume HuggingFace-hosted open models (routed through HF's
# Inference Providers) is primary - Gemini's free tier turned out to be too
# restrictive (~20 requests/day) for how many LLM calls a single CRAG question makes
# (grade every doc + filter every strip). Gemini and Groq's Qwen3 stay wired in
# unchanged as the last two fallback tiers, after every HF model has been tried.
hf_chats = [_build_hf_chat(repo_id) for repo_id in settings.hf_chat_models]

llm = hf_chats[0].with_fallbacks(hf_chats[1:] + [primary_llm, fallback_llm])


def _hf_structured(chat: ChatHuggingFace, schema):
    """HuggingFace's `with_structured_output` only supports a dict/JSON-schema for
    tool-calling - a Pydantic class raises `NotImplementedError` there - so the
    schema is converted to a dict up front and the result coerced back into the
    Pydantic `schema` afterward, to match what Gemini/Groq's chains return.

    Open-weight models occasionally mis-type a tool-call argument (e.g. emitting
    the string "false" instead of the boolean `false`), which the provider rejects
    server-side as a 400 before any response reaches this chain - retrying once
    resolves it most of the time without needing a different model.
    """
    schema_dict = schema.model_json_schema()
    schema_dict["title"] = schema.__name__
    raw_chain = chat.with_structured_output(schema_dict, method="function_calling")
    return (raw_chain | RunnableLambda(lambda d: schema(**d))).with_retry(stop_after_attempt=2)


def structured(schema):
    """Build a Runnable that returns `schema`, looping through every HF model
    first and falling through to Gemini, then Groq, exactly as `llm` above does.

    Qwen's tool-calling occasionally misfires on a structured-output call
    (`tool_use_failed`) even when the service itself is healthy, so the Groq tier
    gets one quick retry of its own on top of taking over for Gemini.
    """
    fallback_structured = fallback_llm.with_structured_output(schema).with_retry(stop_after_attempt=2)
    gemini_structured = primary_llm.with_structured_output(schema).with_fallbacks([fallback_structured])
    hf_structured_chats = [_hf_structured(chat, schema) for chat in hf_chats]
    return hf_structured_chats[0].with_fallbacks(hf_structured_chats[1:] + [gemini_structured])
