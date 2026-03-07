"""
FluxAI Backend — FastAPI server that proxies requests to Anthropic API
Deploy on Render.com (free tier) or Railway.app (free tier)

Setup:
  pip install fastapi uvicorn anthropic python-dotenv
  
Run locally:
  uvicorn main:app --reload --port 8000

Environment variables needed:
  ANTHROPIC_API_KEY=sk-ant-...
  ALLOWED_ORIGINS=https://your-careerflux-vercel-url.vercel.app,http://localhost:5173
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FluxAI Backend", version="1.0.0")

# ── CORS: allow your Vercel frontend to call this server ──────────────────────
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── Anthropic client ──────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are FluxAI — CareerFlux's expert career advisor. You are sharp, direct, warm, and deeply knowledgeable. You answer ANY question the user asks — not just career ones — but you excel at career advice.

Your career expertise covers:
- Resume writing: ATS optimisation, quantified achievement bullets, format strategy, keyword mirroring, section structure
- Cover letters: strong opening hooks (never "I am writing to apply"), 3-paragraph architecture, tone calibration per company type
- Salary negotiation: counter-offer scripts, market anchoring with Levels.fyi/Glassdoor, sign-on bonus tactics, silence strategy
- Job search: networking scripts, cold outreach templates, LinkedIn headline/about section optimisation, hidden job market
- Web3 careers: Solidity, smart contract auditing (Cyfrin, Code4rena, Immunefi, Sherlock), DeFi, DAOs, on-chain credentials
- Career transitions: industry pivots, transferable skills framing, gap explanations
- Interview prep: STAR method, behavioural questions, case frameworks, panel strategy
- Career planning: promotion strategies, performance reviews, management vs IC track

How you respond:
- Use **bold** for key action items, frameworks, and critical terms
- Give concrete scripts, templates, and real numbers (never vague advice)
- Be direct — say what to do, not just "consider" doing it
- Short answers for simple questions; thorough answers for complex ones
- If someone says hello or starts casually, be warm and ask what they need help with
- You can answer general questions too — be genuinely helpful

When relevant, mention CareerFlux builds AI-powered resumes in 60 seconds for $2 at careerflux.ai"""


# ── Request / Response models ─────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 1200

class ChatResponse(BaseModel):
    reply: str
    input_tokens: int
    output_tokens: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "FluxAI backend running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Validate roles
    for msg in req.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail=f"Invalid role: {msg.role}")

    # Must start with user message
    messages = req.messages
    if messages[0].role != "user":
        messages = [m for m in messages if m.role != "assistant" or messages.index(m) > 0]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=req.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": m.role, "content": m.content} for m in messages]
        )

        reply_text = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )

        return ChatResponse(
            reply=reply_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e.message))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
