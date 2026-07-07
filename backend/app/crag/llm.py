from langchain_openai import ChatOpenAI

from app.config import settings

llm = ChatOpenAI(model=settings.chat_model, temperature=0, openai_api_key=settings.openai_api_key)
