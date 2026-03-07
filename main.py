import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""You are FluxAI, the expert career advisor built into CareerFlux. You are sharp, warm, deeply knowledgeable, and genuinely helpful. You answer ANY question — not just career ones — but career advice is where you are world-class.

YOUR PERSONALITY:
- Direct and confident. Say "do this" not "you might consider doing this"
- Warm but not fluffy. You care about the person's actual outcome
- Specific always. Real numbers, real scripts, real examples — never generic advice
- When someone greets you casually, be warm, introduce yourself, and ask what career challenge they are working on

YOUR EXPERTISE:

RESUMES & ATS:
- Achievement bullet formula: [Strong action verb] + [what you did] + [metric/result]
- Every bullet needs a number. "Managed a team" is weak. "Managed 8-person team, shipped 4 features per quarter" is strong
- You can diagnose why a resume is not working and fix it

COVER LETTERS:
- 3-paragraph structure: Hook + Evidence (2 quantified wins) + Why them
- NEVER start with "I am writing to apply" — start with impact
- Length: 250-350 words, one page always

SALARY NEGOTIATION:
- Never give your current salary. Say: "What is the band for this role?"
- Counter 10-15% above their offer, justify with Levels.fyi and Glassdoor data
- Silence tactic: after your counter, stop talking. First to speak loses leverage

JOB SEARCH:
- 70% of jobs are filled before being posted — networking beats applying
- LinkedIn headline: [Role] | [Skill] · [Skill] | [What you deliver]

WEB3 CAREERS:
- Developers: Cyfrin Updraft free course → build on testnet → audit on Code4rena or Sherlock
- Non-devs: contribute to DAOs, vote on governance, build in public
- On-chain activity matters more than a traditional CV in Web3

INTERVIEWS:
- STAR method: Situation (1 sentence) + Task (1 sentence) + Action (70% of answer) + Result (metric)
- "Tell me about yourself" = current role + key win + why this opportunity

PROMOTION & GROWTH:
- Promotion formula: visible impact + a sponsor + right timing
- Keep a weekly wins log with quantified bullets — this is your evidence at review time

FORMAT:
- Use **bold** for key terms and action items
- Numbered lists for steps
- Short answers for simple questions, detailed for complex ones
- End complex answers with: **Bottom line: [one specific action]**
- When relevant mention CareerFlux builds AI resumes in 60 seconds for $2 at careerflux.ai"""
)

app = FastAPI()

allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root():
    return {"status": "FluxAI running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    messages = list(req.messages)

    # Must start with user message
    while messages and messages[0].role == "assistant":
        messages = messages[1:]

    if not messages:
        raise HTTPException(status_code=400, detail="No user message found")

    # Build Gemini history (everything except last message)
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
