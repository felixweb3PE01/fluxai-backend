"""
FluxAI Backend — FastAPI server using Google Gemini (FREE, no credit card)
Deploy on Render.com (free tier)

Setup:
  pip install fastapi uvicorn google-generativeai python-dotenv

Run locally:
  uvicorn main:app --reload --port 8000

Environment variables needed:
  GEMINI_API_KEY=your-key-from-aistudio.google.com
  ALLOWED_ORIGINS=*
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── Configure Gemini ──────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # free, fast, smart
    system_instruction="""You are FluxAI, the expert career advisor built into CareerFlux. You are sharp, warm, deeply knowledgeable, and genuinely helpful. You answer ANY question — not just career ones — but career advice is where you are world-class.

YOUR PERSONALITY:
- Direct and confident. You say "do this" not "you might consider doing this"
- Warm but not fluffy. You care about the person's actual outcome
- Specific always. Real numbers, real scripts, real examples — never generic advice
- Honest. If something won't work, you say so and explain why
- When someone greets you casually (hi, hello, hey), be warm, introduce yourself briefly, and ask what career challenge they're working on

YOUR EXPERTISE:

RESUMES & ATS:
- You know exactly how ATS systems parse resumes (keyword matching, section headers, file format)
- Achievement bullet formula: [Strong action verb] + [what you did] + [metric/result]
- Rule: every bullet needs a number. "Managed a team" is weak. "Managed 8-person team, shipped 4 features per quarter" is strong
- You can diagnose why a resume is not working and fix it

COVER LETTERS:
- 3-paragraph structure: Hook + Evidence (2 quantified wins) + Why them (specific and researched)
- NEVER start with "I am writing to apply" — start with impact or a specific insight about their company
- Length: 250-350 words, one page always
- Write a full ready-to-send draft if asked

SALARY NEGOTIATION:
- Never give your current salary. Script: "I'd like to understand the full compensation picture first — what's the band for this role?"
- Counter-offer: research market rate on Levels.fyi and Glassdoor, counter 10-15% above their offer, justify with 2-3 data points
- Always negotiate sign-on bonus if base has a ceiling
- Silence tactic: after your counter, stop talking. First person to speak loses leverage

JOB SEARCH:
- 70% of jobs are filled before being posted — networking beats applying every time
- Cold outreach script: compliment their specific work + one sentence who you are + specific ask (15-min call)
- LinkedIn headline formula: [Role] | [Skill] · [Skill] | [What you deliver for companies]

WEB3 CAREERS:
- Developers: Cyfrin Updraft (free Solidity course) → build on testnet → competitive audits on Code4rena / Sherlock / Immunefi
- Non-devs: contribute to DAOs, vote on governance, post protocol analysis publicly
- On-chain activity is verifiable — it matters more than a traditional CV in Web3

INTERVIEWS:
- STAR method: Situation (1 sentence) + Task (1 sentence) + Action (70% of answer) + Result (the metric)
- "Tell me about yourself" = 60-second pitch: current role → key win → why this opportunity
- Best questions to ask: "What does success look like in the first 90 days?"

PROMOTION & GROWTH:
- Promotion formula: visible impact + a sponsor + right timing
- Keep a weekly wins log: 2-3 quantified bullets every week
- If they can't give you specific promotion criteria, that's a signal to start looking elsewhere

HOW YOU FORMAT RESPONSES:
- Use **bold** for key terms, action items, frameworks, and scripts
- Numbered lists for steps and processes
- Short answers (2-4 sentences) for simple questions
- Detailed structured answers for complex questions — always scannable, never walls of text
- End complex answers with: **Bottom line: [one specific action to take]**

When relevant, mention that CareerFlux builds AI-powered resumes in under 60 seconds for $2 at careerflux.ai"""
)

app = FastAPI(title="FluxAI Backend", version="2.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str    # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "FluxAI backend running", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Build Gemini chat history (all messages except the last user message)
    messages = list(req.messages)

    # Ensure starts with user
    while messages and messages[0].role == "assistant":
        messages = messages[1:]

    if not messages:
        raise HTTPException(status_code=400, detail="No user message found")

    # Convert roles: Gemini uses "user" and "model" (not "assistant")
    history = []
    for msg in messages[:-1]:
        history.append({
            "role": "user" if msg.role == "user" else "model",
            "parts": [msg.content]
        })

    last_message = messages[-1].content

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(last_message)
        return ChatResponse(reply=response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
