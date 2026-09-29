import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini"

_client = None


def _get_client():
    global _client
    if _client is None:
        if not API_KEY:
            return None
        _client = genai.Client(api_key=API_KEY)
    return _client


def get_learning_recommendations(topic: str) -> str:
    """Return a structured learning path for the given topic."""
    client = _get_client()
    if client is None:
        return "Error: GEMINI_API_KEY is not set in the environment."

    prompt = f"""You are an expert AI tutor. The student wants to learn about: {topic}.

Create a clear, structured learning path that includes:
1. Prerequisites (if any)
2. Beginner level – key concepts and suggested resources
3. Intermediate level – deeper topics and practice ideas
4. Advanced level – challenging projects or further study
5. A suggested order / timeline

Keep the tone encouraging and practical. Use markdown headings and bullet points."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        if response and getattr(response, "text", None):
            return response.text.strip()
        return "Could not generate learning recommendations."
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"