"""Chat model selection. LLM_PROVIDER = groq | gemini | openai | none."""
from ..config import settings


def provider_name() -> str:
    p = settings.LLM_PROVIDER.lower()
    keys = {"groq": settings.GROQ_API_KEY, "gemini": settings.GOOGLE_API_KEY,
            "openai": settings.OPENAI_API_KEY}
    return p if keys.get(p) else "none"


def get_llm():
    """Return a LangChain chat model, or None when no provider/key is configured."""
    p = provider_name()
    if p == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY,
                        temperature=0.1, max_tokens=600, timeout=30, max_retries=1)
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL, google_api_key=settings.GOOGLE_API_KEY,
                                      temperature=0.1, max_output_tokens=600)
    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY,
                          temperature=0.1, max_tokens=600, timeout=30)
    return None
