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


def answer_question_with_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    client = _get_client()
    if client is None:
        return (
            "Error: GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key from "
            "https://aistudio.google.com/app/apikey"
        )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        if response and getattr(response, "text", None):
            return response.text.strip()
        return "No response generated. Please try again."
    except Exception as e:
        return f"Error generating answer: {str(e)}"