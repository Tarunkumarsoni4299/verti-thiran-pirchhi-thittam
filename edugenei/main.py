import os
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import explanation_module
import qna
import summary_module
import quiz_module
import learning_module

load_dotenv()

app = FastAPI(
    title="EduGenie Assistant API",
    description="Intelligent Personalized Learning Assistant powered by Gemini",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field(default="Concept Explainer")


def get_mode_prompt(mode: str, question: str) -> str:
    mode = (mode or "Concept Explainer").strip()

    if mode == "Instant Q&A":
        return (
            "Provide a concise, direct answer with minimal fluff.\n"
            f"Question: {question}"
        )
    if mode == "Summaries & Notes":
        return (
            "Provide a structured summary with clear headings and key takeaways.\n"
            f"Topic / text: {question}"
        )
    if mode == "Interactive Quizzes":
        return f"""Create a 3-question multiple-choice quiz based on: {question}.

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
- Always include exactly 4 options and a short explanation."""
    return (
        "Explain the following concept clearly using simple language and everyday analogies.\n"
        f"Concept: {question}"
    )


async def extract_input_text(request: Request) -> str:
    try:
        data = await request.json()
        return (
            data.get("topic")
            or data.get("prompt")
            or data.get("input_text")
            or data.get("text")
            or data.get("question")
            or ""
        ).strip()
    except Exception:
        return ""


@app.get("/")
async def serve_home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/qa")
@app.post("/api/qa")
@app.post("/qna")
@app.post("/api/qna")
async def answer_question(req: QARequest):
    if not req.question.strip():
        return JSONResponse(
            content={"error": "Please provide a question."},
            status_code=400,
        )

    formatted_prompt = get_mode_prompt(req.mode, req.question)
    answer = qna.answer_question_with_gemini(formatted_prompt)
    return {"answer": answer, "mode": req.mode}


@app.post("/explain")
@app.post("/api/explain")
async def explain_api(request: Request):
    topic = await extract_input_text(request)
    if not topic:
        return JSONResponse(
            content={"error": "Please provide a topic."},
            status_code=400,
        )
    explanation = explanation_module.explain_topic(topic)
    return {
        "topic": topic,
        "explanation": explanation,
        "response": explanation,
        "result": explanation,
    }


@app.post("/summarize")
@app.post("/api/summarize")
async def summarize_api(request: Request):
    text = await extract_input_text(request)
    if not text:
        return JSONResponse(
            content={"error": "Please provide text to summarize."},
            status_code=400,
        )
    summary = summary_module.summarize_text(text)
    return {
        "text": text,
        "summary": summary,
        "response": summary,
        "result": summary,
    }


@app.post("/quiz")
@app.post("/api/quiz")
async def quiz_api(request: Request):
    try:
        text = await extract_input_text(request)
        if not text:
            return JSONResponse(
                content={"error": "Please provide text for the quiz."},
                status_code=400,
            )
        quiz = quiz_module.generate_quiz(text)
        return {"result": quiz}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/learn/recommendations")
@app.post("/api/learn/recommendations")
@app.post("/learning-path")
@app.post("/api/learning-path")
async def learning_recommendation_post(request: Request):
    topic = await extract_input_text(request)
    if not topic:
        return JSONResponse(
            content={"error": "Please provide a topic."},
            status_code=400,
        )
    recommendation = learning_module.get_learning_recommendations(topic)
    return {
        "topic": topic,
        "recommendation": recommendation,
        "response": recommendation,
        "result": recommendation,
    }


@app.get("/learn/recommendations")
@app.get("/api/learn/recommendations")
@app.get("/learning-path")
@app.get("/api/learning-path")
async def learning_recommendation_get(topic: str = Query(..., min_length=1)):
    recommendation = learning_module.get_learning_recommendations(topic)
    return {
        "topic": topic,
        "recommendation": recommendation,
        "response": recommendation,
        "result": recommendation,
    }


@app.get("/health")
async def health():
    key_present = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return {
        "status": "ok",
        "gemini_key_configured": key_present,
    }