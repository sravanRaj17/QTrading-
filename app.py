import re
import random
import math
import sqlite3
import hashlib
import secrets
import io
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from contextlib import contextmanager

# ----------------------------------------------------------------------
# FastAPI & Pydantic imports
# ----------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Try to import Qiskit (Aer simulator). Graceful fallback if missing.
# ----------------------------------------------------------------------
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not installed. Falling back to classical randomness.")

# ----------------------------------------------------------------------
# Try to import librosa for voice analysis. Fallback to mock if missing.
# ----------------------------------------------------------------------
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("librosa/soundfile not installed. Voice analysis will be mocked.")

# ----------------------------------------------------------------------
# Try to import Google Gemini API
# ----------------------------------------------------------------------
try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("google-generativeai not installed. Gemini explanations disabled.")

# ----------------------------------------------------------------------
# Application constants
# ----------------------------------------------------------------------
CAPITAL = 10_000.0          # Trading capital for risk calculation
MAX_RISK_PCT = 0.05          # 5% max risk per trade
SECRET_KEY = secrets.token_hex(32)  # For session middleware
DB_PATH = "trading_psych.db"
GEMINI_API_KEY = os.getenv("AIzaSyBjb2-qopmrG7MXCtiGTnkvIJ_p-4Ah5bE")  # Set this environment variable

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
else:
    GEMINI_MODEL = None

# ----------------------------------------------------------------------
# Database setup
# ----------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                text TEXT,
                amount REAL,
                decision TEXT,
                fear REAL,
                greed REAL,
                confidence REAL,
                discipline INTEGER,
                insight TEXT,
                quantum_bias REAL,
                voice_fear REAL,
                voice_greed REAL,
                voice_confidence REAL,
                ai_explanation TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        # Add column if missing (for existing DBs)
        try:
            conn.execute("ALTER TABLE history ADD COLUMN ai_explanation TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ----------------------------------------------------------------------
# Password hashing (PBKDF2)
# ----------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}${dk.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    salt, stored_hash = hashed.split('$')
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return dk.hex() == stored_hash

# ----------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class AnalysisRequest(BaseModel):
    text: str = Field(..., description="Trader's journal, thoughts, or market commentary")
    amount: float = Field(..., gt=0, description="Proposed trade amount")

class EmotionProbabilities(BaseModel):
    fear: float
    greed: float
    confidence: float

class AnalysisResponse(BaseModel):
    emotion_probabilities: EmotionProbabilities
    discipline_score: int
    behavioral_insight: str
    decision: str
    explanation: str
    quantum_bias: float
    voice_emotions: Optional[EmotionProbabilities] = None
    fused_emotions: Optional[EmotionProbabilities] = None
    ai_insight: Optional[str] = None

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Quantum-Inspired AI Trading Psychology Engine")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# ----------------------------------------------------------------------
# Authentication dependency
# ----------------------------------------------------------------------
async def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    with get_db() as conn:
        user = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

# ----------------------------------------------------------------------
# Helper functions for emotion & behavioral analysis (text)
# ----------------------------------------------------------------------
FEAR_KEYWORDS = [
    "scared", "fear", "panic", "worried", "nervous", "anxious", "uncertain",
    "crash", "loss", "losing", "down", "drop", "tumble", "plunge"
]
GREED_KEYWORDS = [
    "greedy", "greed", "fomo", "miss out", "moon", "pump", "dump",
    "yolo", "all in", "double down", "leverage", "margin"
]
CONFIDENCE_KEYWORDS = [
    "confident", "sure", "strong", "solid", "bullish", "calm", "steady",
    "plan", "strategy", "analysis", "discipline", "patient"
]

def keyword_match_count(text: str, keywords: list) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

def text_emotion_probabilities(text: str) -> Tuple[float, float, float]:
    fear_cnt = keyword_match_count(text, FEAR_KEYWORDS)
    greed_cnt = keyword_match_count(text, GREED_KEYWORDS)
    conf_cnt = keyword_match_count(text, CONFIDENCE_KEYWORDS)
    total = fear_cnt + greed_cnt + conf_cnt
    if total == 0:
        return 0.33, 0.33, 0.34
    return fear_cnt / total, greed_cnt / total, conf_cnt / total

def discipline_score(emotions: Dict[str, float], amount: float) -> int:
    base = emotions["confidence"] * 100 - (emotions["fear"] + emotions["greed"]) * 50
    base = max(0, min(100, base))
    if amount > CAPITAL * MAX_RISK_PCT:
        penalty = 20 + (amount / (CAPITAL * MAX_RISK_PCT) - 1) * 30
        base = max(0, base - penalty)
    return int(round(base))

def behavioral_insight(emotions: Dict[str, float], text: str) -> str:
    text_low = text.lower()
    if "revenge" in text_low:
        return "Revenge trading tendency detected"
    if "overtrading" in text_low or "over trading" in text_low:
        return "Overtrading behavior pattern"
    if emotions["fear"] > 0.5:
        return "Fear-driven hesitation"
    if emotions["greed"] > 0.5:
        return "Greed-driven impulse risk"
    if emotions["confidence"] > 0.5:
        return "Calm, confident state"
    return "Mixed emotional signals"

# ----------------------------------------------------------------------
# Quantum‑inspired bias (with fallback)
# ----------------------------------------------------------------------
def quantum_bias() -> float:
    if not QISKIT_AVAILABLE:
        return random.random()
    try:
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
        return counts.get("1", 0) / 1024
    except Exception as e:
        print(f"Quantum simulation error: {e}. Using fallback.")
        return random.random()

# ----------------------------------------------------------------------
# Decision model
# ----------------------------------------------------------------------
def make_decision(emotions: Dict[str, float], discipline: int, quantum_bias_value: float) -> Tuple[str, str]:
    base_score = emotions["confidence"] * 100 + discipline * 0.5
    penalty = (emotions["fear"] + emotions["greed"]) * 80
    final_score = base_score - penalty
    threshold = 55 + (quantum_bias_value - 0.5) * 10

    if final_score >= threshold:
        decision = "TRADE"
        explanation = "Emotional and risk metrics support trade execution."
    elif final_score >= threshold - 15:
        decision = "CAUTION"
        explanation = "Mixed signals; proceed with reduced size or wait for confirmation."
    else:
        decision = "WAIT"
        explanation = "High emotional distortion or excessive risk; waiting is prudent."
    return decision, explanation

# ----------------------------------------------------------------------
# Voice emotion analysis (librosa) – improved sensitivity
# ----------------------------------------------------------------------
def analyze_voice_emotions(audio_bytes: bytes) -> Tuple[float, float, float]:
    """Extract pitch, energy, tempo and map to fear/greed/confidence with better variation."""
    if not AUDIO_AVAILABLE or not audio_bytes:
        # Return balanced but slightly random fallback
        return (
            random.uniform(0.25, 0.35),
            random.uniform(0.25, 0.35),
            random.uniform(0.30, 0.45)
        )

    try:
        # Load audio from bytes
        data, sr = sf.read(io.BytesIO(audio_bytes))
        # If stereo, convert to mono safely
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        # Ensure sufficient length
        if len(data) < sr * 0.5:  # less than 0.5 seconds
            raise ValueError("Audio too short")

        # Extract features
        # Energy (RMS) – normalize using typical speech range
        rms = librosa.feature.rms(y=data).mean()
        rms_norm = min(rms / 0.3, 1.0)  # typical RMS ~0.05-0.3 for speech

        # Zero-crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y=data).mean()
        zcr_norm = min(zcr / 0.15, 1.0)  # typical ZCR 0.02-0.15

        # Spectral centroid (brightness)
        spec_cent = librosa.feature.spectral_centroid(y=data, sr=sr).mean()
        spec_norm = min(spec_cent / 4000, 1.0)  # typical centroid 500-4000 Hz

        # Tempo
        tempo = librosa.beat.tempo(y=data, sr=sr)[0]
        tempo_norm = min((tempo - 60) / 140, 1.0) if tempo > 60 else 0.0  # 60-200 bpm range

        # Pitch (fundamental frequency) – use piptrack
        pitches, magnitudes = librosa.piptrack(y=data, sr=sr)
        # Average pitch where magnitude > threshold
        pitch_values = pitches[magnitudes > 0.5 * magnitudes.max()] if magnitudes.any() else []
        pitch_mean = pitch_values.mean() if len(pitch_values) > 0 else 150.0
        pitch_norm = min(pitch_mean / 400, 1.0)  # typical pitch 80-400 Hz

        # Heuristic mapping with more dynamic weighting
        # Fear: high pitch + high ZCR + low energy? actually trembling may have fluctuating energy
        fear = (pitch_norm * 0.4 + zcr_norm * 0.3 + (1 - rms_norm) * 0.3)
        # Greed: high energy + high tempo + moderate pitch
        greed = (rms_norm * 0.4 + tempo_norm * 0.4 + pitch_norm * 0.2)
        # Confidence: moderate pitch, steady energy, lower ZCR, higher spectral centroid?
        confidence = ((1 - abs(pitch_norm - 0.5)) * 0.4 + rms_norm * 0.3 + (1 - zcr_norm) * 0.3)

        # Add small random noise to avoid identical outputs (entropy)
        fear += random.uniform(-0.03, 0.03)
        greed += random.uniform(-0.03, 0.03)
        confidence += random.uniform(-0.03, 0.03)

        # Ensure non-negative and normalize
        fear = max(0.01, fear)
        greed = max(0.01, greed)
        confidence = max(0.01, confidence)
        total = fear + greed + confidence
        fear /= total
        greed /= total
        confidence /= total

        return fear, greed, confidence

    except Exception as e:
        print(f"Voice analysis error: {e}")
        # Fallback with slight randomness
        return (
            random.uniform(0.25, 0.35),
            random.uniform(0.25, 0.35),
            random.uniform(0.30, 0.45)
        )

def fuse_emotions(text_emotions: Tuple[float, float, float],
                  voice_emotions: Tuple[float, float, float],
                  voice_weight: float = 0.4) -> Tuple[float, float, float]:
    """Weighted fusion of text and voice emotion probabilities."""
    tf, tg, tc = text_emotions
    vf, vg, vc = voice_emotions
    w = voice_weight
    fused_fear = tf * (1 - w) + vf * w
    fused_greed = tg * (1 - w) + vg * w
    fused_conf = tc * (1 - w) + vc * w
    total = fused_fear + fused_greed + fused_conf
    return fused_fear / total, fused_greed / total, fused_conf / total

# ----------------------------------------------------------------------
# Gemini AI explanation generator
# ----------------------------------------------------------------------
async def generate_gemini_insight(
    text: str,
    emotions: Dict[str, float],
    discipline: int,
    insight: str,
    decision: str,
    amount: float,
    quantum_bias: float,
    voice_emotions: Optional[Dict[str, float]] = None
) -> Optional[str]:
    """Use Gemini to produce a professional, actionable explanation."""
    if not GEMINI_AVAILABLE or not GEMINI_MODEL:
        return None

    prompt = f"""You are an expert trading psychologist and a senior financial advisor. 
Given the following analysis of a trader's current state, provide a concise but insightful explanation (3-5 sentences) that includes:
- Interpretation of the emotional profile and its likely impact on trading decisions.
- One specific, actionable suggestion to improve discipline or risk management.
- A note on how the quantum bias (randomness factor) might be influencing the decision threshold.

Trader's Journal Entry: "{text}"
Proposed Trade Amount: ${amount:,.2f}
Emotion Probabilities (Fear/Greed/Confidence): {emotions['fear']:.2f}/{emotions['greed']:.2f}/{emotions['confidence']:.2f}
""" + (f"Voice Analysis Emotions: Fear {voice_emotions['fear']:.2f}, Greed {voice_emotions['greed']:.2f}, Confidence {voice_emotions['confidence']:.2f}\n" if voice_emotions else "") + f"""
Discipline Score: {discipline}/100
Behavioral Insight: {insight}
Final Decision: {decision}
Quantum Bias: {quantum_bias:.3f} (range 0-1, where 0.5 is neutral)

Please respond with only the explanation text, no extra formatting or markdown."""

    try:
        response = await GEMINI_MODEL.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

# ----------------------------------------------------------------------
# Auth endpoints
# ----------------------------------------------------------------------
@app.post("/register")
async def register(user: UserRegister, request: Request):
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (user.username,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        hashed = hash_password(user.password)
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (user.username, hashed))
        conn.commit()
    return {"message": "Registration successful"}

@app.post("/login")
async def login(user: UserLogin, request: Request):
    with get_db() as conn:
        db_user = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (user.username,)).fetchone()
        if not db_user or not verify_password(user.password, db_user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        request.session["user_id"] = db_user["id"]
    return {"message": "Login successful"}

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")

# ----------------------------------------------------------------------
# Protected analysis endpoint (text only)
# ----------------------------------------------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_trade(
    request: AnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    fear, greed, conf = text_emotion_probabilities(request.text)
    emotions = {"fear": fear, "greed": greed, "confidence": conf}
    disc = discipline_score(emotions, request.amount)
    insight = behavioral_insight(emotions, request.text)
    qb = quantum_bias()
    decision, explanation = make_decision(emotions, disc, qb)

    # Save to history
    with get_db() as conn:
        conn.execute("""
            INSERT INTO history (user_id, text, amount, decision, fear, greed, confidence, discipline, insight, quantum_bias)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (current_user["id"], request.text, request.amount, decision, fear, greed, conf, disc, insight, qb))
        conn.commit()

    # Try Gemini
    ai_insight = await generate_gemini_insight(
        text=request.text,
        emotions=emotions,
        discipline=disc,
        insight=insight,
        decision=decision,
        amount=request.amount,
        quantum_bias=qb
    )

    return AnalysisResponse(
        emotion_probabilities=EmotionProbabilities(fear=fear, greed=greed, confidence=conf),
        discipline_score=disc,
        behavioral_insight=insight,
        decision=decision,
        explanation=explanation,
        quantum_bias=qb,
        ai_insight=ai_insight
    )

# ----------------------------------------------------------------------
# Voice-only analysis endpoint
# ----------------------------------------------------------------------
@app.post("/analyze_voice")
async def analyze_voice(audio: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    audio_bytes = await audio.read()
    fear, greed, conf = analyze_voice_emotions(audio_bytes)
    return {"fear": fear, "greed": greed, "confidence": conf}

# ----------------------------------------------------------------------
# Full multimodal analysis (text + voice)
# ----------------------------------------------------------------------
@app.post("/analyze_full")
async def analyze_full(
    text: str = Form(...),
    amount: float = Form(...),
    audio: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    # Text emotions
    tf, tg, tc = text_emotion_probabilities(text)
    
    # Voice emotions if audio provided and non-empty
    voice_emotions = None
    vf = vg = vc = None
    if audio and audio.filename:
        audio_bytes = await audio.read()
        if len(audio_bytes) > 0:
            vf, vg, vc = analyze_voice_emotions(audio_bytes)
            voice_emotions = EmotionProbabilities(fear=vf, greed=vg, confidence=vc)

    # Fuse emotions
    if voice_emotions:
        fused = fuse_emotions((tf, tg, tc), (vf, vg, vc))
        fear, greed, conf = fused
    else:
        fear, greed, conf = tf, tg, tc

    emotions = {"fear": fear, "greed": greed, "confidence": conf}
    disc = discipline_score(emotions, amount)
    insight = behavioral_insight(emotions, text)
    qb = quantum_bias()
    decision, explanation = make_decision(emotions, disc, qb)

    # Save to history (including ai_explanation later)
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO history (user_id, text, amount, decision, fear, greed, confidence, discipline, insight, quantum_bias, voice_fear, voice_greed, voice_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (current_user["id"], text, amount, decision, fear, greed, conf, disc, insight, qb,
              voice_emotions.fear if voice_emotions else None,
              voice_emotions.greed if voice_emotions else None,
              voice_emotions.confidence if voice_emotions else None))
        history_id = cursor.lastrowid
        conn.commit()

    # Generate Gemini insight
    voice_dict = {"fear": vf, "greed": vg, "confidence": vc} if voice_emotions else None
    ai_insight = await generate_gemini_insight(
        text=text,
        emotions=emotions,
        discipline=disc,
        insight=insight,
        decision=decision,
        amount=amount,
        quantum_bias=qb,
        voice_emotions=voice_dict
    )

    # Update history with AI insight
    if ai_insight:
        with get_db() as conn:
            conn.execute("UPDATE history SET ai_explanation = ? WHERE id = ?", (ai_insight, history_id))
            conn.commit()

    return {
        "emotion_probabilities": {"fear": fear, "greed": greed, "confidence": conf},
        "discipline_score": disc,
        "behavioral_insight": insight,
        "decision": decision,
        "explanation": explanation,
        "quantum_bias": qb,
        "voice_emotions": voice_emotions.dict() if voice_emotions else None,
        "text_emotions": {"fear": tf, "greed": tg, "confidence": tc},
        "ai_insight": ai_insight
    }

# ----------------------------------------------------------------------
# Get user history
# ----------------------------------------------------------------------
@app.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT timestamp, text, amount, decision, fear, greed, confidence, discipline, insight, ai_explanation
            FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20
        """, (current_user["id"],)).fetchall()
    return [dict(row) for row in rows]

# ----------------------------------------------------------------------
# Frontend templates (embedded HTML) – Enhanced UI with Gemini display
# ----------------------------------------------------------------------

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Psychology | Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            background: radial-gradient(circle at 20% 30%, #0f172a, #020617);
            font-family: 'Inter', sans-serif;
            color: #e2e8f0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        #loginParticles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }
        .card {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border-radius: 32px;
            padding: 2.5rem;
            width: 100%;
            max-width: 420px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 40px rgba(0,0,0,0.6), 0 0 0 1px rgba(59, 130, 246, 0.1) inset;
            position: relative;
            z-index: 1;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 30px 50px rgba(0,0,0,0.7), 0 0 0 1px rgba(59, 130, 246, 0.2) inset;
        }
        h1 {
            font-weight: 600;
            font-size: 2rem;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #e0eaff, #9bb9ff);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 0.5rem;
            position: relative;
        }
        h1::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 0;
            width: 40px;
            height: 2px;
            background: #3b82f6;
            border-radius: 2px;
        }
        .sub {
            color: #94a3b8;
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }
        label {
            display: block;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 0.4rem;
        }
        input {
            width: 100%;
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(4px);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 0.8rem 1rem;
            font-size: 1rem;
            color: #f1f5f9;
            margin-bottom: 1.2rem;
            transition: border 0.2s, box-shadow 0.2s;
        }
        input:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px #1e3a8a40;
        }
        button {
            width: 100%;
            background: #1e3a8a;
            border: none;
            color: white;
            font-weight: 600;
            padding: 0.9rem;
            border-radius: 40px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
            border: 1px solid #3b82f6;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        button:hover {
            background: #2563eb;
            transform: scale(1.02);
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
        }
        .toggle-link {
            text-align: center;
            margin-top: 1.2rem;
            color: #94a3b8;
        }
        .toggle-link a {
            color: #60a5fa;
            text-decoration: none;
            transition: color 0.2s;
        }
        .toggle-link a:hover {
            color: #93c5fd;
        }
        .error {
            background: #450a0a40;
            color: #fca5a5;
            padding: 0.7rem;
            border-radius: 16px;
            margin-bottom: 1rem;
            font-size: 0.85rem;
            border: 1px solid #7f1d1d;
            display: none;
            backdrop-filter: blur(4px);
        }
    </style>
</head>
<body>
    <canvas id="loginParticles"></canvas>
    <div class="card">
        <h1>Quantum Psychology</h1>
        <div class="sub">Sign in to access your trading behavior dashboard</div>
        <div id="errorBox" class="error"></div>
        <form id="authForm">
            <label>Username</label>
            <input type="text" id="username" autocomplete="username" required>
            <label>Password</label>
            <input type="password" id="password" autocomplete="current-password" required>
            <button type="submit" id="submitBtn">Sign In</button>
        </form>
        <div class="toggle-link">
            <span id="toggleText">Don't have an account?</span> <a href="#" id="toggleLink">Register</a>
        </div>
    </div>
<script>
    // Simple particle background for login page
    const canvas = document.getElementById('loginParticles');
    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];
    const PARTICLE_COUNT = 50;

    function initParticles() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push({
                x: Math.random(),
                y: Math.random(),
                vx: (Math.random() - 0.5) * 0.002,
                vy: (Math.random() - 0.5) * 0.002,
                size: Math.random() * 2 + 1
            });
        }
    }

    function resizeCanvas() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
    }

    function drawParticles() {
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#3b82f6';
        for (let p of particles) {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = 1;
            if (p.x > 1) p.x = 0;
            if (p.y < 0) p.y = 1;
            if (p.y > 1) p.y = 0;
            
            const px = p.x * width;
            const py = p.y * height;
            ctx.beginPath();
            ctx.arc(px, py, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(59, 130, 246, ${0.2 + Math.sin(Date.now()*0.001 + p.size)*0.1})`;
            ctx.fill();
        }
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.08)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i < particles.length; i++) {
            for (let j = i+1; j < particles.length; j++) {
                const dx = (particles[i].x - particles[j].x) * width;
                const dy = (particles[i].y - particles[j].y) * height;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x * width, particles[i].y * height);
                    ctx.lineTo(particles[j].x * width, particles[j].y * height);
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(drawParticles);
    }

    window.addEventListener('resize', () => {
        resizeCanvas();
    });
    resizeCanvas();
    initParticles();
    drawParticles();

    // Auth logic
    const form = document.getElementById('authForm');
    const username = document.getElementById('username');
    const password = document.getElementById('password');
    const submitBtn = document.getElementById('submitBtn');
    const errorBox = document.getElementById('errorBox');
    const toggleLink = document.getElementById('toggleLink');
    const toggleText = document.getElementById('toggleText');
    let mode = 'login';

    toggleLink.addEventListener('click', (e) => {
        e.preventDefault();
        mode = mode === 'login' ? 'register' : 'login';
        submitBtn.textContent = mode === 'login' ? 'Sign In' : 'Register';
        toggleText.textContent = mode === 'login' ? "Don't have an account?" : "Already have an account?";
        toggleLink.textContent = mode === 'login' ? 'Register' : 'Sign In';
        errorBox.style.display = 'none';
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        errorBox.style.display = 'none';
        const payload = { username: username.value, password: password.value };
        const endpoint = mode === 'login' ? '/login' : '/register';
        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Authentication failed');
            }
            if (mode === 'register') {
                const loginRes = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!loginRes.ok) throw new Error('Auto-login failed');
            }
            window.location.href = '/dashboard';
        } catch (err) {
            errorBox.textContent = err.message;
            errorBox.style.display = 'block';
        }
    });
</script>
</body>
</html>
"""

DASHBOARD_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trading Psychology | Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: 'Inter', sans-serif;
            color: #eef2ff;
            min-height: 100vh;
            padding: 2rem 1.5rem;
            position: relative;
            overflow-x: hidden;
            background: #05080c;
        }
        #particleCanvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
            opacity: 0.9;
        }
        .dashboard {
            max-width: 1600px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }
        .card {
            background: rgba(18, 25, 40, 0.55);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border-radius: 28px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 20px 35px -8px rgba(0,0,0,0.7), 0 0 0 1px rgba(59, 130, 246, 0.1) inset;
            transition: transform 0.3s cubic-bezier(0.2, 0, 0, 1), box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px) scale(1.002);
            box-shadow: 0 30px 45px -10px rgba(0,0,0,0.8), 0 0 0 1px rgba(59, 130, 246, 0.25) inset;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .title-section h1 {
            font-weight: 600;
            font-size: 2rem;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #e0eaff, #9bb9ff);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            position: relative;
            display: inline-block;
        }
        .title-section h1::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 0;
            width: 60px;
            height: 3px;
            background: #3b82f6;
            border-radius: 3px;
        }
        .subhead {
            color: #8fa0bf;
            margin-top: 0.75rem;
            font-size: 0.9rem;
            border-left: 3px solid #3b82f6;
            padding-left: 1rem;
        }
        .user-area {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        .logout-btn {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(4px);
            border: 1px solid #475569;
            color: #cbd5e1;
            padding: 0.5rem 1.5rem;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        .logout-btn:hover {
            background: #1e293b;
            border-color: #64748b;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .grid-2col {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.8rem;
        }
        .card-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9bb1d0;
            margin-bottom: 1.2rem;
            position: relative;
        }
        .card-title i {
            color: #3b82f6;
            font-size: 1rem;
            width: 20px;
        }
        .card-title::before {
            content: '';
            width: 4px;
            height: 16px;
            background: #3b82f6;
            border-radius: 4px;
            margin-right: 8px;
        }
        .progress-bg {
            background: #10161f;
            border-radius: 12px;
            height: 8px;
            overflow: hidden;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
        }
        .progress-fill {
            height: 100%;
            border-radius: 12px;
            transition: width 0.6s cubic-bezier(0.2, 0.9, 0.4, 1);
            box-shadow: 0 0 8px currentColor;
        }
        .fill-fear { background: linear-gradient(90deg, #f97316, #ef4444); }
        .fill-greed { background: linear-gradient(90deg, #facc15, #eab308); }
        .fill-conf { background: linear-gradient(90deg, #22c55e, #10b981); }
        .emotion-row { margin-bottom: 1rem; }
        .emotion-label { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.3rem; font-weight: 500; }
        .risk-level {
            display: inline-block;
            padding: 0.25rem 1rem;
            border-radius: 40px;
            font-weight: 600;
            font-size: 0.8rem;
            letter-spacing: 0.3px;
            backdrop-filter: blur(4px);
        }
        .risk-low { background: #16653430; color: #4ade80; border: 1px solid #22c55e40; }
        .risk-medium { background: #854d0e30; color: #facc15; border: 1px solid #eab30840; }
        .risk-high { background: #7f1d1d30; color: #f87171; border: 1px solid #ef444440; }
        .decision-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(12, 17, 27, 0.7);
            backdrop-filter: blur(12px);
            border-radius: 28px;
            padding: 1rem 1.8rem;
            margin: 1.5rem 0;
            border-left: 6px solid;
            transition: box-shadow 0.4s ease, border-left-color 0.3s;
        }
        .decision-text {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .decision-glow-trade { box-shadow: 0 0 30px rgba(16, 185, 129, 0.5); }
        .decision-glow-caution { box-shadow: 0 0 30px rgba(245, 158, 11, 0.5); }
        .decision-glow-wait { box-shadow: 0 0 30px rgba(239, 68, 68, 0.5); }
        .input-panel {
            display: flex;
            flex-wrap: wrap;
            gap: 1.2rem;
            margin-bottom: 2rem;
            background: rgba(15, 22, 34, 0.6);
            backdrop-filter: blur(16px);
            padding: 1.5rem;
            border-radius: 28px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .input-group { flex: 1 1 240px; }
        label {
            display: block;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #99aacf;
            letter-spacing: 0.03em;
        }
        textarea, input {
            width: 100%;
            background: rgba(12, 17, 30, 0.7);
            backdrop-filter: blur(4px);
            border: 1px solid #293548;
            border-radius: 20px;
            padding: 0.8rem 1rem;
            font-size: 0.9rem;
            color: #f0f4ff;
            font-family: inherit;
            transition: border 0.2s, box-shadow 0.2s;
        }
        textarea:focus, input:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px #1e3a8a40;
        }
        button {
            background: #1e3a8a;
            border: none;
            color: white;
            font-weight: 600;
            padding: 0.8rem 2rem;
            border-radius: 40px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #3b82f6;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
            font-size: 0.95rem;
        }
        button:hover:not(:disabled) {
            background: #2563eb;
            transform: scale(1.02);
            box-shadow: 0 6px 18px rgba(59, 130, 246, 0.5);
        }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .voice-panel {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .recording-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .pulse {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ef4444;
            box-shadow: 0 0 15px #ef4444;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
        .waveform {
            width: 100%;
            height: 60px;
            background: #0c111e;
            border-radius: 12px;
            margin-top: 0.5rem;
            border: 1px solid #1f2a3e;
        }
        .breakdown-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
        .breakdown-grid div:nth-child(odd) { color: #9bb1d0; }
        .breakdown-grid div:nth-child(even) { font-weight: 600; color: #e2e8f0; }
        .history-list {
            list-style: none;
            max-height: 240px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .history-list li {
            background: rgba(11, 16, 26, 0.6);
            backdrop-filter: blur(4px);
            margin-bottom: 0.7rem;
            padding: 0.9rem 1.2rem;
            border-radius: 20px;
            font-size: 0.85rem;
            display: flex;
            justify-content: space-between;
            border-left: 4px solid;
            transition: transform 0.2s;
        }
        .history-list li:hover {
            transform: translateX(4px);
            background: rgba(20, 30, 48, 0.7);
        }
        .footer {
            margin-top: 2.5rem;
            text-align: center;
            font-size: 0.75rem;
            color: #4f658d;
            letter-spacing: 0.5px;
        }
        .data-updated { animation: softFlash 0.6s ease-out; }
        @keyframes softFlash {
            0% { opacity: 0.7; }
            50% { opacity: 1; background: rgba(59, 130, 246, 0.1); }
            100% { opacity: 1; }
        }
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .inline-spinner { margin-left: 8px; }
        .ai-insight-box {
            background: rgba(59, 130, 246, 0.1);
            border-radius: 18px;
            padding: 1.2rem;
            margin-top: 1.5rem;
            border-left: 4px solid #3b82f6;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .ai-insight-box i { color: #60a5fa; margin-right: 6px; }
    </style>
</head>
<body>
    <canvas id="particleCanvas"></canvas>
    <div class="dashboard">
        <div class="header">
            <div class="title-section">
                <h1>Quantum-Inspired Psychology Engine</h1>
                <div class="subhead">Emotion analytics · Risk discipline · Voice analysis · Gemini AI insights</div>
            </div>
            <div class="user-area">
                <span id="usernameDisplay" style="font-weight:500;"></span>
                <button class="logout-btn" onclick="logout()"><i class="fas fa-sign-out-alt"></i> Sign Out</button>
            </div>
        </div>

        <!-- Input Panel -->
        <div class="input-panel">
            <div class="input-group" style="flex:2;">
                <label><i class="fas fa-pen"></i> Trading journal / notes</label>
                <textarea id="tradeText" placeholder="Describe your emotional state, market view...">Market looks shaky but I'm confident in this setup. No FOMO.</textarea>
            </div>
            <div class="input-group">
                <label><i class="fas fa-dollar-sign"></i> Trade amount (USD)</label>
                <input type="number" id="tradeAmount" value="2500" min="1" step="100">
            </div>
            <div class="input-group">
                <label><i class="fas fa-microphone"></i> Voice Analysis (optional)</label>
                <div class="voice-panel">
                    <button id="recordBtn" type="button"><i class="fas fa-circle"></i> Start Recording</button>
                    <button id="stopBtn" type="button" disabled><i class="fas fa-stop"></i> Stop</button>
                    <button id="playBtn" type="button" disabled><i class="fas fa-play"></i> Play</button>
                    <span id="recordingStatus" class="recording-indicator"></span>
                </div>
                <canvas id="waveformCanvas" class="waveform" width="400" height="60"></canvas>
                <input type="hidden" id="audioData" value="">
            </div>
            <div class="input-group" style="display: flex; align-items: flex-end;">
                <button id="analyzeBtn"><i class="fas fa-brain"></i> Analyze Psychology</button>
            </div>
        </div>
        <div id="errorContainer" style="color:#f97316; background: rgba(44, 26, 26, 0.5); backdrop-filter: blur(4px); padding:0.7rem 1.2rem; border-radius:20px; margin-bottom:1rem; display:none; border:1px solid #7f1d1d;"></div>

        <!-- Emotion Cards -->
        <div class="grid-2col">
            <div class="card" id="emotionCard">
                <div class="card-title"><i class="fas fa-chart-pie"></i> Emotion Spectrum</div>
                <div id="textEmotions"></div>
                <div id="voiceEmotions" style="margin-top:1rem; border-top:1px solid #1e2a3e; padding-top:1rem;"></div>
                <div class="emotion-row"><div class="emotion-label"><span><i class="fas fa-frown" style="color:#ef4444;"></i> Fear</span><span id="fearVal">0.00</span></div><div class="progress-bg"><div id="fearBar" class="progress-fill fill-fear" style="width:0%"></div></div></div>
                <div class="emotion-row"><div class="emotion-label"><span><i class="fas fa-coins" style="color:#eab308;"></i> Greed</span><span id="greedVal">0.00</span></div><div class="progress-bg"><div id="greedBar" class="progress-fill fill-greed" style="width:0%"></div></div></div>
                <div class="emotion-row"><div class="emotion-label"><span><i class="fas fa-smile" style="color:#10b981;"></i> Confidence</span><span id="confVal">0.00</span></div><div class="progress-bg"><div id="confBar" class="progress-fill fill-conf" style="width:0%"></div></div></div>
            </div>
            <div class="card">
                <div class="card-title"><i class="fas fa-shield-alt"></i> Discipline & Risk</div>
                <div style="display:flex; justify-content:center;"><canvas id="disciplineCanvas" width="120" height="120"></canvas></div>
                <div style="display:flex; justify-content:space-between; margin: 1rem 0 0.25rem;"><span><i class="fas fa-exclamation-triangle"></i> Risk exposure</span><span id="riskPercentLabel">0%</span></div>
                <div class="progress-bg"><div id="riskBarFill" class="progress-fill" style="width:0%; background:#ea580c;"></div></div>
                <div style="margin-top:10px;"><span id="riskQualitative" class="risk-level">—</span></div>
            </div>
        </div>

        <!-- Decision Breakdown & AI Insight -->
        <div class="grid-2col">
            <div class="card">
                <div class="card-title"><i class="fas fa-lightbulb"></i> Behavioral Insight</div>
                <div style="font-size:1.1rem; font-weight:500; margin-bottom: 1rem;" id="insightText">—</div>
                <div class="breakdown-grid">
                    <div><i class="fas fa-keyboard"></i> Text Fear/Greed/Conf</div><div id="textContrib">—</div>
                    <div><i class="fas fa-microphone-alt"></i> Voice Fear/Greed/Conf</div><div id="voiceContrib">—</div>
                    <div><i class="fas fa-merge"></i> Fused Emotions</div><div id="fusedContrib">—</div>
                    <div><i class="fas fa-gavel"></i> Discipline Score</div><div id="discContrib">—</div>
                    <div><i class="fas fa-atom"></i> Quantum Bias</div><div id="qbContrib">—</div>
                </div>
                <div style="margin-top:1.5rem;">
                    <div style="display:flex; justify-content:space-between;"><span><i class="fas fa-chart-line"></i> Decision Confidence</span><span id="decisionConfidenceValue">—</span></div>
                    <div class="progress-bg"><div id="decisionConfidenceBar" class="progress-fill" style="width:0%; background:#8b5cf6;"></div></div>
                </div>
                <!-- AI Insight Display -->
                <div id="aiInsightBox" class="ai-insight-box" style="display:none;">
                    <i class="fas fa-robot"></i> <strong>Gemini AI Insight:</strong>
                    <div id="aiInsightText" style="margin-top:8px;"></div>
                </div>
            </div>
            <div class="card">
                <div class="card-title"><i class="fas fa-robot"></i> Decision Engine</div>
                <div class="decision-banner" id="decisionBanner" style="margin:0.5rem 0;">
                    <span class="decision-text" id="decisionLabel">WAIT</span>
                    <span id="explanationText" style="max-width:60%;">Analyze to get recommendation</span>
                </div>
                <div class="progress-bg" style="margin:12px 0;"><div id="qbBar" class="progress-fill" style="width:0%; background:#a78bfa;"></div></div>
                <div style="display:flex; justify-content:space-between;"><span><i class="fas fa-dice"></i> Quantum Bias:</span><span id="qbValue">—</span></div>
                <div style="margin-top:1.2rem; font-size:0.9rem; color:#b0c4de;" id="detailedExplanation"></div>
            </div>
        </div>

        <!-- History -->
        <div class="card">
            <div class="card-title"><i class="fas fa-history"></i> Behavioral History</div>
            <ul id="historyList" class="history-list"><li>Loading history...</li></ul>
        </div>
        <div class="footer">Quantum circuit (AerSimulator) · Real-time risk analytics · Voice emotion fusion · Gemini AI</div>
    </div>

<script>
    (function(){
        // ----- PARTICLE BACKGROUND (enhanced) -----
        const canvas = document.getElementById('particleCanvas');
        const ctx = canvas.getContext('2d');
        let width, height;
        let particles = [];
        const PARTICLE_COUNT = 70;
        let mouseX = 0.5, mouseY = 0.5;
        
        function initParticles() {
            particles = [];
            for (let i = 0; i < PARTICLE_COUNT; i++) {
                particles.push({
                    x: Math.random(),
                    y: Math.random(),
                    vx: (Math.random() - 0.5) * 0.0015,
                    vy: (Math.random() - 0.5) * 0.0015,
                    size: Math.random() * 2.5 + 1,
                    baseSize: Math.random() * 2.5 + 1
                });
            }
        }
        
        function resizeCanvas() {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width;
            canvas.height = height;
        }
        
        function drawParticles() {
            ctx.clearRect(0, 0, width, height);
            for (let p of particles) {
                p.x += p.vx + (mouseX - 0.5) * 0.0001;
                p.y += p.vy + (mouseY - 0.5) * 0.0001;
                if (p.x < 0) p.x = 1;
                if (p.x > 1) p.x = 0;
                if (p.y < 0) p.y = 1;
                if (p.y > 1) p.y = 0;
                p.size = p.baseSize + Math.sin(Date.now() * 0.002 + p.x) * 0.5;
                const px = p.x * width;
                const py = p.y * height;
                const gradient = ctx.createRadialGradient(px, py, 0, px, py, p.size * 2);
                gradient.addColorStop(0, `rgba(59, 130, 246, ${0.25 + Math.sin(Date.now()*0.003)*0.1})`);
                gradient.addColorStop(1, 'rgba(139, 92, 246, 0)');
                ctx.beginPath();
                ctx.arc(px, py, p.size * 1.8, 0, Math.PI * 2);
                ctx.fillStyle = gradient;
                ctx.fill();
            }
            ctx.strokeStyle = 'rgba(99, 143, 255, 0.06)';
            ctx.lineWidth = 0.6;
            for (let i = 0; i < particles.length; i++) {
                for (let j = i+1; j < particles.length; j++) {
                    const dx = (particles[i].x - particles[j].x) * width;
                    const dy = (particles[i].y - particles[j].y) * height;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < 130) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x * width, particles[i].y * height);
                        ctx.lineTo(particles[j].x * width, particles[j].y * height);
                        ctx.strokeStyle = `rgba(59, 130, 246, ${0.08 * (1 - dist/130)})`;
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(drawParticles);
        }
        
        window.addEventListener('resize', () => resizeCanvas());
        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX / window.innerWidth;
            mouseY = e.clientY / window.innerHeight;
        });
        resizeCanvas();
        initParticles();
        drawParticles();

        // ----- Dashboard Logic -----
        const CAPITAL = 10000, MAX_RISK_PCT = 0.05;
        let audioChunks = [], mediaRecorder, audioContext, analyser, audioBlob, audioUrl;
        const waveCanvas = document.getElementById('waveformCanvas'), ctxCanvas = waveCanvas.getContext('2d');
        const gaugeCanvas = document.getElementById('disciplineCanvas'), gaugeCtx = gaugeCanvas.getContext('2d');
        
        async function fetchUser() {
            const res = await fetch('/history');
            if (!res.ok) { window.location.href = '/'; return; }
            const history = await res.json();
            document.getElementById('usernameDisplay').textContent = 'Trader';
            renderHistory(history);
        }
        function renderHistory(history) {
            const list = document.getElementById('historyList');
            if (!history.length) { list.innerHTML = '<li>No history yet</li>'; return; }
            list.innerHTML = history.slice(0,8).map(h => {
                let decisionIcon = '';
                if (h.decision === 'TRADE') decisionIcon = '<i class="fas fa-check-circle" style="color:#10b981;"></i>';
                else if (h.decision === 'CAUTION') decisionIcon = '<i class="fas fa-exclamation-circle" style="color:#f59e0b;"></i>';
                else decisionIcon = '<i class="fas fa-times-circle" style="color:#ef4444;"></i>';
                return `
                <li style="border-left-color: ${h.decision==='TRADE'?'#10b981':h.decision==='CAUTION'?'#f59e0b':'#ef4444'}">
                    <div>${decisionIcon} <strong>${h.decision}</strong> — ${h.insight || '—'}</div>
                    <div><i class="far fa-clock"></i> ${new Date(h.timestamp).toLocaleTimeString()}</div>
                </li>`;
            }).join('');
        }
        function logout() {
            window.location.href = '/logout';
        }

        // Voice recording
        document.getElementById('recordBtn').onclick = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audioUrl = URL.createObjectURL(audioBlob);
                    document.getElementById('playBtn').disabled = false;
                    document.getElementById('audioData').value = 'recorded';
                    visualizeWaveform();
                };
                mediaRecorder.start();
                document.getElementById('recordBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('recordingStatus').innerHTML = '<span class="pulse"></span> Recording...';
                setupAnalyser(stream);
            } catch (e) {
                alert('Microphone access denied or not available. Voice analysis will be skipped.');
            }
        };
        document.getElementById('stopBtn').onclick = () => {
            if (mediaRecorder) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
            }
            document.getElementById('recordBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('recordingStatus').innerHTML = '';
        };
        document.getElementById('playBtn').onclick = () => {
            if (audioUrl) new Audio(audioUrl).play();
        };
        function setupAnalyser(stream) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            drawWaveform();
        }
        function drawWaveform() {
            if (!analyser) return;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            function draw() {
                requestAnimationFrame(draw);
                analyser.getByteFrequencyData(dataArray);
                ctxCanvas.clearRect(0,0,waveCanvas.width,waveCanvas.height);
                const barWidth = waveCanvas.width / bufferLength;
                let x = 0;
                for(let i=0; i<bufferLength; i++) {
                    const h = dataArray[i] / 2;
                    ctxCanvas.fillStyle = '#3b82f6';
                    ctxCanvas.fillRect(x, waveCanvas.height-h, barWidth, h);
                    x += barWidth+1;
                }
            }
            draw();
        }
        function visualizeWaveform() { /* placeholder */ }

        // Analysis
        document.getElementById('analyzeBtn').onclick = async () => {
            const btn = document.getElementById('analyzeBtn');
            const originalContent = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Analyzing...';
            
            const text = document.getElementById('tradeText').value;
            const amount = parseFloat(document.getElementById('tradeAmount').value);
            const formData = new FormData();
            formData.append('text', text);
            formData.append('amount', amount);
            if (audioBlob) {
                formData.append('audio', audioBlob, 'recording.wav');
            }
            try {
                const res = await fetch('/analyze_full', { method:'POST', body: formData });
                if (!res.ok) throw new Error('Analysis failed');
                const data = await res.json();
                updateUI(data, amount);
                fetchUser();
                document.querySelectorAll('.card').forEach(c => c.classList.add('data-updated'));
                setTimeout(() => document.querySelectorAll('.card').forEach(c => c.classList.remove('data-updated')), 600);
                document.getElementById('errorContainer').style.display = 'none';
            } catch(e) {
                document.getElementById('errorContainer').style.display='block';
                document.getElementById('errorContainer').textContent = e.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = originalContent;
            }
        };

        function updateUI(data, amount) {
            const emo = data.emotion_probabilities;
            document.getElementById('fearVal').textContent = emo.fear.toFixed(2);
            document.getElementById('greedVal').textContent = emo.greed.toFixed(2);
            document.getElementById('confVal').textContent = emo.confidence.toFixed(2);
            document.getElementById('fearBar').style.width = (emo.fear*100)+'%';
            document.getElementById('greedBar').style.width = (emo.greed*100)+'%';
            document.getElementById('confBar').style.width = (emo.confidence*100)+'%';
            
            drawGauge(data.discipline_score);
            updateRiskUI(amount);
            document.getElementById('insightText').textContent = data.behavioral_insight;
            document.getElementById('decisionLabel').textContent = data.decision;
            document.getElementById('explanationText').textContent = data.explanation;
            document.getElementById('qbValue').textContent = data.quantum_bias.toFixed(4);
            document.getElementById('qbBar').style.width = (data.quantum_bias*100)+'%';
            
            const confValue = emo.confidence * 100;
            const disc = data.discipline_score;
            const qb = data.quantum_bias;
            let decisionConf = (confValue * 0.6 + disc * 0.4);
            if (data.decision === 'WAIT') decisionConf = Math.min(decisionConf, 40);
            else if (data.decision === 'CAUTION') decisionConf = Math.min(decisionConf, 70);
            decisionConf = Math.min(100, Math.max(0, decisionConf));
            document.getElementById('decisionConfidenceValue').textContent = decisionConf.toFixed(1) + '%';
            document.getElementById('decisionConfidenceBar').style.width = decisionConf + '%';
            
            const banner = document.getElementById('decisionBanner');
            banner.style.borderLeftColor = data.decision==='TRADE'?'#10b981':data.decision==='CAUTION'?'#f59e0b':'#ef4444';
            banner.classList.remove('decision-glow-trade', 'decision-glow-caution', 'decision-glow-wait');
            if (data.decision === 'TRADE') banner.classList.add('decision-glow-trade');
            else if (data.decision === 'CAUTION') banner.classList.add('decision-glow-caution');
            else banner.classList.add('decision-glow-wait');
            
            const textE = data.text_emotions || emo;
            document.getElementById('textContrib').textContent = `${textE.fear.toFixed(2)}/${textE.greed.toFixed(2)}/${textE.confidence.toFixed(2)}`;
            if (data.voice_emotions) {
                const v = data.voice_emotions;
                document.getElementById('voiceContrib').textContent = `${v.fear.toFixed(2)}/${v.greed.toFixed(2)}/${v.confidence.toFixed(2)}`;
            } else {
                document.getElementById('voiceContrib').textContent = '—';
            }
            document.getElementById('fusedContrib').textContent = `${emo.fear.toFixed(2)}/${emo.greed.toFixed(2)}/${emo.confidence.toFixed(2)}`;
            document.getElementById('discContrib').textContent = data.discipline_score;
            document.getElementById('qbContrib').textContent = data.quantum_bias.toFixed(3);
            
            // AI Insight
            const aiBox = document.getElementById('aiInsightBox');
            const aiText = document.getElementById('aiInsightText');
            if (data.ai_insight) {
                aiText.textContent = data.ai_insight;
                aiBox.style.display = 'block';
            } else {
                aiBox.style.display = 'none';
            }

            let detail = '';
            if (emo.fear > 0.4) detail += 'Elevated fear detected. ';
            if (emo.greed > 0.4) detail += 'Greed impulse present. ';
            if (data.discipline_score < 40) detail += 'Discipline is low. ';
            if (data.quantum_bias > 0.7) detail += 'Quantum bias suggests heightened randomness. ';
            else if (data.quantum_bias < 0.3) detail += 'Quantum bias indicates stable environment. ';
            if (!detail) detail = 'Emotional state appears balanced. ';
            document.getElementById('detailedExplanation').textContent = detail + data.explanation;
        }

        function drawGauge(score) {
            gaugeCtx.clearRect(0,0,120,120);
            gaugeCtx.beginPath();
            gaugeCtx.arc(60,60,48,0,2*Math.PI);
            gaugeCtx.strokeStyle='#1f2a3e'; gaugeCtx.lineWidth=10; gaugeCtx.stroke();
            gaugeCtx.beginPath();
            gaugeCtx.arc(60,60,48,-0.5*Math.PI, -0.5*Math.PI + (score/100)*2*Math.PI);
            gaugeCtx.strokeStyle='#3b82f6'; gaugeCtx.lineWidth=10; gaugeCtx.stroke();
            gaugeCtx.shadowColor = '#3b82f6'; gaugeCtx.shadowBlur = 10;
            gaugeCtx.stroke();
            gaugeCtx.shadowBlur = 0;
            gaugeCtx.font='bold 22px Inter'; gaugeCtx.fillStyle='#f0f4ff';
            gaugeCtx.textAlign='center'; gaugeCtx.textBaseline='middle';
            gaugeCtx.fillText(score,60,60);
        }
        function updateRiskUI(amount) {
            const pct = (amount/CAPITAL)*100;
            document.getElementById('riskPercentLabel').textContent = pct.toFixed(1)+'%';
            const bar = document.getElementById('riskBarFill');
            bar.style.width = Math.min(100, (amount/(CAPITAL*MAX_RISK_PCT))*100)+'%';
            const qual = document.getElementById('riskQualitative');
            if (amount <= CAPITAL*0.02) { qual.innerHTML = '<i class="fas fa-check-circle"></i> Low risk'; qual.className='risk-level risk-low'; }
            else if (amount <= CAPITAL*MAX_RISK_PCT) { qual.innerHTML = '<i class="fas fa-exclamation-circle"></i> Medium risk'; qual.className='risk-level risk-medium'; }
            else { qual.innerHTML = '<i class="fas fa-times-circle"></i> High risk'; qual.className='risk-level risk-high'; }
        }

        fetchUser();
    })();
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Routes for serving HTML
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/dashboard")
    return HTMLResponse(content=LOGIN_PAGE)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/")
    return HTMLResponse(content=DASHBOARD_PAGE)

# ----------------------------------------------------------------------
# Run with: uvicorn app:app --reload
# ----------------------------------------------------------------------
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
