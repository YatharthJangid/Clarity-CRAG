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


import json
import re


def _extract_schema(res, schema):
    if res is None:
        raise ValueError(f"Model returned None for {schema.__name__}")
    if isinstance(res, schema):
        return res
    if isinstance(res, dict):
        return schema(**res)

    text = ""
    if hasattr(res, "content"):
        text = str(res.content)
    elif isinstance(res, str):
        text = res

    if text:
        # Strip markdown fences if present (```json ... ```)
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return schema(**data)

    raise ValueError(f"Could not parse {schema.__name__} from model output: {res}")


def _build_model_chain(model, schema):
    def _call_and_parse(prompt_input):
        # 1. Try structured output (tool-calling)
        try:
            res = model.with_structured_output(schema).invoke(prompt_input)
            if res is not None:
                if isinstance(res, schema):
                    return res
                if isinstance(res, dict):
                    return schema(**res)
        except Exception:
            pass

        # 2. Fallback to raw generation and regex JSON extraction
        raw = model.invoke(prompt_input)
        return _extract_schema(raw, schema)

    return RunnableLambda(_call_and_parse)


def structured(schema):
    """Build a Runnable that returns `schema`, trying Gemini first and falling through to Groq.
    Handles both tool calls and direct JSON text output seamlessly.
    """
    gemini_chain = _build_model_chain(primary_llm, schema)
    groq_chain = _build_model_chain(fallback_llm, schema)
    return gemini_chain.with_fallbacks([groq_chain])
