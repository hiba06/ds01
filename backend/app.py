# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sys
import os
import json
import re

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from llm.openrouter_helper import ask_llm

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --------------------------------------------------
# APP INIT
# --------------------------------------------------
app = FastAPI(title="Crypto Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# COIN CONFIGURATION
# --------------------------------------------------
COINS = {
    "btc": {
        "name": "Bitcoin",
        "chart": "btc_chart_data.csv",
        "prediction": "btc_prediction.csv",
    },
    "eth": {
        "name": "Ethereum",
        "chart": "eth_chart_data.csv",
        "prediction": "eth_prediction.csv",
    },
    "sol": {
        "name": "Solana",
        "chart": "sol_chart_data.csv",
        "prediction": "sol_prediction.csv",
    },
}

# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------
def load_chart_data(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path)

    if "timestamp" not in df.columns or len(df.columns) < 2:
        return None, None

    value_col = df.columns[1]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", value_col])

    return df, value_col


def format_chart_response(df: pd.DataFrame, value_col: str):
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype("int64") // 10**9

    return [
        {"time": int(t), "value": float(v)}
        for t, v in zip(df["timestamp"], df[value_col])
    ]


def load_prediction(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        return {"predicted_normalized_price": None}

    df = pd.read_csv(path)

    if "predicted_normalized_price" not in df.columns:
        return {"predicted_normalized_price": None}

    return {
        "predicted_normalized_price":
            float(df["predicted_normalized_price"].iloc[0])
    }


def safe_llm_json_parse(raw: str):
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return json.loads(match.group())
    except Exception:
        return None


def generate_llm_insight(coin_name: str, df: pd.DataFrame, value_col: str):
    recent = df.tail(30)

    if len(recent) < 5:
        return default_llm_response(
            "Insufficient recent data for analysis."
        )

    price_change = recent[value_col].iloc[-1] - recent[value_col].iloc[0]
    volatility = recent[value_col].std()

    prompt = f"""
You are a professional crypto market analyst.

Analyze recent normalized price behavior of {coin_name}.

Metrics:
- Net price change: {price_change:.6f}
- Volatility: {volatility:.6f}

Return STRICT JSON only:
{{
  "market_regime": "Trending | Ranging | Volatile",
  "momentum": "Bullish | Bearish | Neutral",
  "volatility_state": "Low | Medium | High",
  "risk_outlook": "Conservative | Balanced | Aggressive",
  "key_insight": "ONE short sentence",
  "caution": "ONE short cautionary note"
}}

Rules:
- Neutral tone
- No predictions
- No markdown
"""

    raw = ask_llm(prompt)
    parsed = safe_llm_json_parse(raw)

    if parsed is None:
        return default_llm_response(
            "Market conditions could not be clearly classified."
        )

    return parsed


def default_llm_response(message: str):
    return {
        "market_regime": "Unknown",
        "momentum": "Neutral",
        "volatility_state": "Unknown",
        "risk_outlook": "Balanced",
        "key_insight": message,
        "caution": "Analysis confidence is limited."
    }

# --------------------------------------------------
# ROOT
# --------------------------------------------------
@app.get("/")
def home():
    return {"message": "Crypto Analytics API Running"}

# --------------------------------------------------
# GENERIC COIN ROUTES
# --------------------------------------------------
@app.get("/{coin}/chart")
def coin_chart(coin: str):
    coin = coin.lower()

    if coin not in COINS:
        return []

    df, value_col = load_chart_data(COINS[coin]["chart"])

    if df is None:
        return []

    return format_chart_response(df, value_col)


@app.get("/{coin}/prediction")
def coin_prediction(coin: str):
    coin = coin.lower()

    if coin not in COINS:
        return {"predicted_normalized_price": None}

    return load_prediction(COINS[coin]["prediction"])


@app.get("/llm/{coin}-summary")
def coin_llm_summary(coin: str):
    coin = coin.lower()

    if coin not in COINS:
        return {"coin": coin.upper(), "insight": None}

    df, value_col = load_chart_data(COINS[coin]["chart"])

    if df is None:
        return {"coin": coin.upper(), "insight": None}

    insight = generate_llm_insight(
        COINS[coin]["name"],
        df,
        value_col
    )

    return {
        "coin": coin.upper(),
        "insight": insight
    }

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
