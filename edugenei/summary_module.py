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


def summarize_text(text: str) -> str:
    """Summarize the given text in clear, simple language."""
    client = _get_client()
    if client is None:
        return "Error: GEMINI_API_KEY is not set in the environment."

    try:
        prompt = (
            "Summarize the following text in simple, clear language. "
            "Use short paragraphs and bullet points for key takeaways when helpful:\n\n"
            f"{text}"
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        if response and getattr(response, "text", None):
            return response.text.strip()
        return "No summary generated."
    except Exception as e:
        return f"Error generating summary: {str(e)}"