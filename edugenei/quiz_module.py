import os
import json
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


def generate_quiz(text: str) -> list:
    """
    Generate a 3-question multiple-choice quiz.
    Returns a list of dicts with: question, options, answer (0-based index), explanation.
    """
    client = _get_client()
    if client is None:
        return [{
            "question": "API key missing",
            "options": [
                "Set GEMINI_API_KEY in .env",
                "Get a free key from Google AI Studio",
                "Restart the server",
                "Try again",
            ],
            "answer": 0,
            "explanation": "Add your Gemini API key to the .env file and restart.",
        }]

    prompt = f"""Create a 3-question multiple-choice quiz based on: {text}.

Return ONLY a raw JSON array. No markdown, no backticks, no extra text.
Exact format:
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": 0,
    "explanation": "Brief explanation why the correct option is right."
  }}
]
Rules:
- "answer" must be the 0-based index of the correct option.
- Always include 4 options and a short explanation.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        raw = (response.text or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Expected a non-empty JSON array")

        for item in data:
            if not all(k in item for k in ("question", "options", "answer", "explanation")):
                raise ValueError("Quiz item missing required keys")
            if not isinstance(item["options"], list) or len(item["options"]) < 2:
                raise ValueError("Each question needs at least 2 options")

        return data

    except Exception as e:
        return [{
            "question": f"Could not generate quiz: {str(e)}",
            "options": [
                "Try again with a clearer topic",
                "Check your API key",
                "Use a different mode",
                "Restart the server",
            ],
            "answer": 0,
            "explanation": "There was a problem contacting the AI or parsing the response.",
        }]