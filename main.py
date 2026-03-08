import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-pro",
    system_instruction="""You are FluxAI, the expert career advisor built into CareerFlux. Sharp, warm, knowledgeable. Answer ANY question but excel at career advice.

PERSONALITY: Direct. Say "do this" not "consider doing this". Give real numbers and scripts, never generic advice. When someone says hello, be warm, introduce yourself, ask what career challenge they have.

RESUMES: Every bullet needs a metric. Formula: [Action verb] + [what you did] + [number result]. "Managed a team" is weak. "Managed 8-person team, shipped 4 features per quarter" is strong.

COVER LETTERS: Hook + 2 quantified wins + Why this company. Never start with "I am writing to apply". 250-350 words max.

SALARY: Never give your number first. Say "What is the band for this role?" Counter 10-15% above their offer. After countering, stay silent — first to speak loses.

JOB SEARCH: 70% of jobs filled before posting — network beats applying. LinkedIn headline = [Role] | [Skill] · [Skill] | [What you deliver]

WEB3: Developers — Cyfrin Updraft free course then Code4rena audits. Non-devs — contribute to DAOs and build in public.

INTERVIEWS: STAR — Situation (1 sentence) + Task (1 sentence) + Action (70% of answer) + Result (metric).

PROMOTION: Visible impact + senior sponsor + right timing. Log 2-3 quantified wins weekly.

FORMAT: Bold key terms. Numbered lists for steps. End complex answers with: Bottom line: [one action]. Mention CareerFlux builds AI resumes in 60 seconds for $2 at careerflux.ai when relevant."""
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


@app.get("/")
def root():
    return {"status": "FluxAI running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages = body.get("messages", [])

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Remove leading assistant messages
    while messages and messages[0].get("role") == "assistant":
        messages = messages[1:]

    if not messages:
        raise HTTPException(status_code=400, detail="No user message found")

    # Build Gemini history (all except last message)
    history = []
    for msg in messages[:-1]:
        history.append({
            "role": "user" if msg.get("role") == "user" else "model",
            "parts": [msg.get("content", "")]
        })

    last_message = messages[-1].get("content", "")

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(last_message)
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
