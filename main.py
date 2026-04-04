import os
from dotenv import load_dotenv
load_dotenv()

import json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()


class EmailRequest(BaseModel):
    subject: str
    sender: str
    body: str
    links: list
    attachments: list


@app.get("/health")
def health():
    return {"status": "ok", "service": "JARVIS-X AI Scoring", "model": "groq/llama-3.1-8b-instant"}


@app.get("/test-groq")
def test_groq():
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say hello in one word"}],
            temperature=0.1,
            max_tokens=200
        )
        text = response.choices[0].message.content.strip()
        print(f"Groq test response: {text}")
        return {"response": text, "status": "ok"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.post("/analyze")
def analyze_email(email: EmailRequest):
    try:
        prompt = f"""You are an email security expert. Analyze this email and return ONLY a JSON object with no markdown, no backticks, no explanation.

Email:
Subject: {email.subject}
Sender: {email.sender}
Body: {email.body}
Links: {email.links}
Attachments: {email.attachments}

Return exactly this format:
{{"score": 85, "threatLevel": "HIGH", "reason": "one sentence"}}

Rules:
- score 0-40 = LOW
- score 41-60 = MEDIUM
- score 61-100 = HIGH"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )

        raw = response.choices[0].message.content.strip()
        print(f"Groq raw response: {raw}")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        return result

    except Exception as e:
        print(f"Analysis error: {type(e).__name__}: {str(e)}")
        return {"score": 0, "threatLevel": "LOW", "reason": "Analysis failed"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
