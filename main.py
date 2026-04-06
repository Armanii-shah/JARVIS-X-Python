import os
import json
import re
import unicodedata
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="JARVIS-X AI Scoring", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class EmailRequest(BaseModel):
    subject: str = Field(default="", max_length=500)
    sender: str = Field(default="", max_length=500)
    body: str = Field(default="", max_length=50000)
    links: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)

    @field_validator("links", "attachments", mode="before")
    @classmethod
    def coerce_list(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            return [str(v)]
        return v


def map_threat_level(score: int) -> str:
    if score <= 40:
        return "LOW"
    if score <= 60:
        return "MEDIUM"
    return "HIGH"


def extract_json(raw: str) -> dict:
    """Strip markdown fences and extract the first valid JSON object."""
    # Remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract first {...} block as fallback
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"No valid JSON found in response: {raw[:200]}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "JARVIS-X AI Scoring",
        "model": "groq/llama-3.1-8b-instant",
        "groq_key_set": bool(GROQ_API_KEY),
    }


@app.post("/analyze")
def analyze_email(email: EmailRequest):
    subject = email.subject.strip() or "(No Subject)"
    sender = email.sender.strip() or "(Unknown Sender)"
    body_preview = email.body[:500].strip()

    # Remove non-ASCII characters that can break JSON parsing
    body_preview = body_preview.encode('ascii', 'ignore').decode('ascii')
    subject = subject.encode('ascii', 'ignore').decode('ascii')
    sender = sender.encode('ascii', 'ignore').decode('ascii')

    # Remove extra whitespace
    body_preview = ' '.join(body_preview.split())

    links = email.links[:20]       # cap to prevent oversized prompts
    attachments = email.attachments[:20]

    prompt = f"""You are a strict email security expert. Score this email from 0-100.

Email:
Subject: {subject}
Sender: {sender}
Body preview: {body_preview}
Links: {links}
Attachments: {attachments}

SCORING BANDS — pick the MOST specific one:

0-10: Completely safe. Legitimate sender domain, no links, personal/professional content.
Examples: GitHub notifications, Google/Microsoft official emails, personal emails from contacts.

11-25: Low risk. Known brand newsletter, tracking links only, no urgency, real domain.
Examples: MongoDB newsletter, LinkedIn job alert, Notion updates, YouTube recommendations.

26-45: Slightly suspicious. Unknown sender but no malicious content, mild promotional language.
Examples: Cold outreach email, unknown newsletter, generic promotional email.

46-60: Moderate risk. Urgency language OR suspicious link OR slightly fake domain. Only ONE red flag.
Examples: "Your account needs attention" with real domain, mildly suspicious link, vague threat.

61-75: High risk. Multiple red flags: urgency + suspicious domain, OR fake domain mimicking real brand.
Examples: "netfl1x-billing.xyz", "paypa1-secure.com", urgency + suspicious link.

76-88: Very high risk. Clear phishing: fake brand domain + urgency + suspicious link OR dangerous attachment.
Examples: Netflix/PayPal phishing with .xyz domain + urgency + one .exe attachment.

89-100: CRITICAL. Multiple attack vectors combined: fake government/bank + multiple .exe/.bat/.cmd/.vbs/.ps1 attachments + ransom/arrest threats + crypto payment requests.
Examples: FIA arrest notice + 10 malware files + bitcoin payment demand.

CRITICAL RULES:
- Legitimate known services (Google, GitHub, LinkedIn, MongoDB, Notion) = ALWAYS 0-25
- Same threat level emails must get DIFFERENT scores (vary by ±5-10 points based on specifics)
- Count red flags: each one adds 10-15 points
- Real company domain = automatically below 30
- Fake domain (.xyz, .ml, .tk mimicking real brand) = minimum 61
- Each dangerous attachment (.exe, .bat, .cmd, .vbs, .ps1, .jar, .msi) = +8 points
- Ransom/arrest/government impersonation = +20 points minimum
- Multiple attack vectors = 89-100 ONLY

Return ONLY valid JSON:
{{"score": 45, "threatLevel": "MEDIUM", "reason": "Specific reason with exact threat indicators found"}}

threatLevel mapping:
- 0-40 = "LOW"
- 41-60 = "MEDIUM"
- 61-100 = "HIGH"

Never give the same score to two different emails unless they are identical threats."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
            timeout=15,
        )
    except Exception as e:
        print(f"[Groq] API error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"AI service error: {type(e).__name__}")

    if not response.choices:
        print("[Groq] Empty choices returned")
        raise HTTPException(status_code=502, detail="AI service returned empty response")

    raw = response.choices[0].message.content
    if not raw:
        print("[Groq] Empty content in response")
        raise HTTPException(status_code=502, detail="AI service returned empty content")

    raw = raw.strip()
    print(f"[Groq] Raw response: {raw}")

    try:
        result = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[Groq] JSON parse error: {e}")
        raise HTTPException(status_code=502, detail="AI service returned malformed JSON")

    # Clamp score to 0-100
    try:
        score = int(result.get("score", 50))
    except (TypeError, ValueError):
        score = 50
    score = max(0, min(100, score))

    # Always derive threatLevel from score — never trust AI's mapping
    threat_level = map_threat_level(score)

    reason = str(result.get("reason", "No reason provided")).strip() or "No reason provided"

    return {
        "score": score,
        "threatLevel": threat_level,
        "reason": reason,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
