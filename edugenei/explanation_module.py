import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-flash-latest"

_client = None


def _get_client():
    global _client
    if _client is None:
        if not API_KEY:
            return None
        _client = genai.Client(api_key=API_KEY)
    return _client


def explain_topic(topic: str) -> str:
    """Explain a concept clearly with simple analogies (uses Gemini)."""
    client = _get_client()
    if client is None:
        return "Error: GEMINI_API_KEY is not set in the environment."

    try:
        prompt = (
            f"Explain the concept of '{topic}' in a simple and clear way "
            "suitable for a school or early college student. "
            "Use everyday analogies and short paragraphs. "
            "Avoid unnecessary jargon."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        if response and getattr(response, "text", None):
            return response.text.strip()
        return "No explanation generated."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"