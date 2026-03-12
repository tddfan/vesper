"""
Vesper v5.0 — Intelligence Engine (Multi-Currency Apex)
"""

from __future__ import annotations

import asyncio
import calendar
import datetime
import hashlib
import io
import json
import time
import traceback
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple
import copy
import csv
import re

try:
    import anthropic as _anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

try:
    import google.generativeai as _gemini
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False
    print("[Vesper] WARNING: 'google-generativeai' SDK not found. Gemini integration will be disabled.")

import numpy as np
import pandas as pd
import yfinance as yf
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv(override=True)

async def send_telegram_alert(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"})
    except Exception as e:
        print(f"[Telegram Error] {e}")

def _daily_return_zscore(df: pd.DataFrame, window: int = 30) -> float:
    """Compute z-score of the latest daily return relative to its own trailing volatility.
    Returns how many standard deviations today's move is from the rolling mean.
    """
    if df.empty or 'Close' not in df.columns or len(df) < max(window, 5):
        return 0.0
    rets = df['Close'].pct_change().dropna()
    if len(rets) < window:
        return 0.0
    trailing = rets.iloc[-(window + 1):-1]  # last N days *before* today
    mu = float(trailing.mean())
    sigma = float(trailing.std())
    if sigma < 1e-8:
        return 0.0
    latest = float(rets.iloc[-1])
    return (latest - mu) / sigma


async def get_lead_lag_signals(holdings: Dict[str, float], ohlc_data: Dict[str, pd.DataFrame], live_vix: float = None) -> List[Dict]:
    """
    Vesper v6.0 Lead-Lag Engine — Relative Volatility Model.
    Copper (HG=F) leads Industrial/India.
    BTC-USD leads Tech/Risk-on.

    Trigger: Leader move exceeds +2σ (relative to its own trailing vol)
    AND the lag asset moved < 0.5σ. This eliminates false triggers from
    BTC's naturally high volatility or Copper's noisy days.
    """
    alerts = []
    # Compute leader z-scores (not raw % change)
    leader_data: Dict[str, Dict] = {}
    for l_tk in ("HG=F", "BTC-USD"):
        if l_tk in ohlc_data:
            df = ohlc_data[l_tk]
            if len(df) >= 5:
                raw_chg = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
                z = _daily_return_zscore(df)
                leader_data[l_tk] = {"change_pct": raw_chg, "zscore": z}

    # Get VIX for regime-adjusted stops — prefer live quote
    if live_vix is not None and live_vix > 0:
        vix_val = live_vix
    elif "^VIX" in ohlc_data and len(ohlc_data["^VIX"]) >= 1:
        vix_val = float(ohlc_data["^VIX"]["Close"].iloc[-1])
    else:
        vix_val = 15.0

    for tk, qty in holdings.items():
        clean_tk = tk.split('_')[0].split('.')[0].upper()
        # Industrial/India proxies
        is_ind = any(x in clean_tk for x in ["IIND", "XMME", "GLDX", "COPX", "EPIC"])
        # Tech/Risk-on proxies
        is_risk = any(x in clean_tk for x in ["EQQQ", "QQQ", "SMH", "SOXX", "BTC", "COIN", "ARKK"])

        target_df = ohlc_data.get(tk.split('_')[0], pd.DataFrame())
        if target_df.empty or len(target_df) < 5: continue
        target_price = float(target_df['Close'].iloc[-1])
        target_chg = (target_price / target_df['Close'].iloc[-2] - 1) * 100
        target_z = _daily_return_zscore(target_df)

        leader_info = None
        leader_sym = ""
        if is_ind and "HG=F" in leader_data:
            leader_info = leader_data["HG=F"]
            leader_sym = "Copper (HG=F)"
        elif is_risk and "BTC-USD" in leader_data:
            leader_info = leader_data["BTC-USD"]
            leader_sym = "Bitcoin (BTC-USD)"

        if not leader_info:
            continue

        leader_z = leader_info["zscore"]
        leader_chg = leader_info["change_pct"]

        # Trigger: leader > +2σ AND target < 0.5σ (relative to own vol)
        if leader_z >= 2.0 and abs(target_z) < 0.5:
            # Conviction scales with leader z-score: 2σ→85%, 3σ→92%, 4σ+→96%
            conviction = min(96, round(80 + leader_z * 5))
            atr = _compute_atr(target_df) if len(target_df) >= 15 else (0.02 * target_price)
            vix_adj = 1 + (vix_val / 100)
            sl = round(target_price - (atr * vix_adj), 2)
            tp = round(target_price + (3 * atr), 2)
            risk = round(target_price - sl, 2)
            reward = round(tp - target_price, 2)
            rr = round(reward / risk, 1) if risk > 0 else 0
            alerts.append({
                "type": "LEAD-LAG",
                "symbol": tk,
                "title": f"Statistical Lag-Drift ({leader_z:.1f}σ Leader Move)",
                "rationale": (
                    f"{leader_sym} moved {leader_chg:+.1f}% ({leader_z:.1f}σ vs trailing 30d vol) "
                    f"while {tk} lagged at {target_chg:+.2f}% ({target_z:.1f}σ). "
                    f"Cross-asset mean-reversion model flags convergence at >{conviction}% confidence."
                ),
                "conviction": conviction,
                "entry_price": round(target_price, 2),
                "stop_loss": sl,
                "take_profit": tp,
                "risk_reward": f"1:{rr}",
                "atr_raw": round(atr, 4),
                "vix_adjustment": round(vix_adj, 2),
                "leader_zscore": round(leader_z, 2),
                "target_zscore": round(target_z, 2),
            })
    return alerts

def get_institutional_flow(holdings: Dict[str, float], ohlc_data: Dict[str, pd.DataFrame] = None, vix: float = 15.0) -> List[Dict]:
    """
    Vesper v6.0 Institutional Flow — Volume-Based Detection.

    Detects abnormal institutional activity using real OHLCV data:
    1. Volume Ratio: Today's volume vs 20-day SMA. Ratio > 1.5 = unusual.
    2. Smart Money Accumulation: Volume ratio > 1.5 AND Price Change > 0.
    3. Distribution/Selling: Volume ratio > 1.5 AND Price Change < 0.

    Returns the same JSON structure expected by the frontend with dynamic rationales.
    """
    alerts = []
    vix_adj = 1 + (vix / 100)

    for tk in holdings:
        api_tk = tk.split('_')[0]
        hist_df = ohlc_data.get(api_tk, pd.DataFrame()) if ohlc_data else pd.DataFrame()
        if hist_df.empty or 'Volume' not in hist_df.columns or len(hist_df) < 21:
            continue

        vol_series = hist_df['Volume'].astype(float)
        today_vol = vol_series.iloc[-1]
        avg_vol_20 = float(vol_series.iloc[-21:-1].mean())

        if avg_vol_20 < 1:
            continue

        vol_ratio = today_vol / avg_vol_20
        if vol_ratio <= 1.5:  # User threshold: 1.5x
            continue

        # Price change today
        entry_price = float(hist_df['Close'].iloc[-1])
        prev_close = float(hist_df['Close'].iloc[-2])
        day_chg_pct = (entry_price / prev_close - 1) * 100 if prev_close > 0 else 0

        # Classify signal type based on user rules
        vol_pct_above = round((vol_ratio - 1) * 100)
        if day_chg_pct > 0:
            signal_type = "Smart Money Accumulation"
            signal_title = "Institutional Flow — Smart Money Accumulation"
            rationale = (
                f"{api_tk} volume is {vol_pct_above}% above its 20-day average "
                f"while price gained {day_chg_pct:+.2f}%. "
                f"Suggests institutional entry and high-conviction accumulation."
            )
            # Score 80+ as requested
            conviction = min(98, 80 + int((vol_ratio - 1.5) * 5))
        else:
            signal_type = "Distribution/Selling"
            signal_title = "Institutional Flow — Distribution/Selling"
            rationale = (
                f"{api_tk} volume is {vol_pct_above}% above average on a {day_chg_pct:+.2f}% price drop. "
                f"Heavy institutional selling or distribution detected."
            )
            conviction = min(95, 75 + int((vol_ratio - 1.5) * 5))

        atr = _compute_atr(hist_df) if len(hist_df) >= 15 else (0.02 * entry_price)
        sl = round(entry_price - (atr * vix_adj), 2)
        tp = round(entry_price + (3 * atr), 2)
        risk = round(entry_price - sl, 2)
        reward = round(tp - entry_price, 2)
        rr = round(reward / risk, 1) if risk > 0 else 0

        alerts.append({
            "type": "VOLUME-ANOMALY",
            "symbol": tk,
            "title": signal_title,
            "signal_subtype": signal_type,
            "rationale": rationale,
            "conviction": conviction,
            "entry_price": round(entry_price, 2),
            "stop_loss": sl,
            "take_profit": tp,
            "risk_reward": f"1:{rr}",
            "atr_raw": round(atr, 4),
            "vix_adjustment": round(vix_adj, 2),
            "volume_ratio": round(vol_ratio, 2),
            "today_volume": int(today_vol),
            "avg_volume_20d": int(avg_vol_20),
        })

    alerts.sort(key=lambda x: x["conviction"], reverse=True)
    return alerts[:5]


# ── Firebase Admin (Firestore persistence) ────────────────────────────────────
try:
    from firebase_admin import firestore as _fs, auth as _fb_auth
    import firebase_admin as _fb_admin
    try:
        _fb_admin.get_app()        # already initialized (e.g. Cloud Functions wrapper ran first)
    except ValueError:
        _fb_admin.initialize_app()  # first initialization
    _FS_CLIENT = _fs.client()
    _FIREBASE_OK = True
except Exception:
    _FS_CLIENT = None          # type: ignore[assignment]
    _FIREBASE_OK = False

HF_TOKEN = os.getenv("HF_TOKEN")
FINBERT_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

_sentiment_cache: Dict[str, Dict[str, Any]] = {} 
_executor = ThreadPoolExecutor(max_workers=15)

CACHE_TTL = 3600
RISK_FREE_RATE = 0.04   # UK 10-yr gilt proxy

_CIO_SYSTEM_PROMPT = """\
You are the Chief Investment Officer (CIO) of an elite quantitative macro hedge fund. \
Your mandate is ruthless capital preservation and absolute return generation. \
You are cold, analytical, and entirely unsentimental about assets. \
You view global news solely through the lens of risk-adjusted returns, duration exposure, \
and supply-demand imbalances.

You are powering the "Vesper v5.0 Institutional Apex" wealth management terminal.

Every day you receive:
1. The client's current portfolio holdings (Tickers and GBP values).
2. The client's uninvested cash balance in GBP.
3. The latest global macroeconomic and geopolitical news headlines.

YOUR TASK:
Analyse the news and portfolio metrics to generate exactly 10 "Executive Directives" for the client. 
1. Sort these directives by descending priority: CRITICAL (immediate threat/opportunity) > HIGH > MEDIUM > ROUTINE.
2. Ensure they are written in clear, jargon-free layman language.
3. If the client has significant uninvested cash, the top directives must address specific deployment steps.
4. Also adjust the broader "Tactical Blueprint" weights (Growth, Core, Defensive) based on the current macro regime.

RULES & HEURISTICS:
- Inflation/Hot CPI: Trim long-duration growth/tech; buy short-duration bonds or commodities.
- Geopolitical Escalation: Buy defence, oil, or gold. Trim exposed regional equities.
- Market Panics: If fundamentals are intact, buy the dip on high-quality assets (RSI oversold).
- Euphoria / RSI Overbought: Trim positions and take profit before mean-reversion.
- ONLY suggest trades for tickers the client already holds OR these standard institutional ETFs: \
SGLN.L (Gold), TN28.L (UK Gilts), EQQQ.L (Nasdaq), NRGG.L (Oil & Gas E&P), CMOP.L (Commodities), \
IGLT.L (Gilts), HMSO.L (Gold).

OUTPUT FORMAT:
Respond with ONLY a single, strict, valid JSON object — no markdown, no commentary outside it.

{
  "master_verdict": {
    "summary": "2-3 sentences providing an executive summary of the current regime and primary portfolio focus.",
    "directives": [
      {
        "priority": "CRITICAL | HIGH | MEDIUM | ROUTINE",
        "action": "Clear, bold layman instruction (e.g., 'Deploy £5k into Gold')",
        "timeframe": "Immediate (0-4h) | Tactical (24-72h) | Strategic (1-2 weeks)",
        "rationale": "One-sentence layman explanation of why."
      }
    ]
  },
  "tactical_desk": [
    {
      "icon": "🛢️",
      "severity": "GEO-RISK",
      "time_horizon": "72h",
      "event": "Short headline summarising the risk",
      "action": "Buy",
      "ticker": "NRGG.L",
      "amount": 5000,
      "exit_strategy": {
        "stop_loss": 23.45,
        "take_profit": 28.50,
        "rationale": "Dynamic ATR-based exit strategy. Stop = Price - 2*ATR; TP = Price + 3*ATR."
      },
      "rationale": "Institutional explanation of why this trade protects capital or captures alpha."
    }
  ],
  "tactical_blueprint": {
    "target": {
      "Growth": 0.35,
      "Core": 0.45,
      "Defensive": 0.20
    },
    "rationale": "+5% Defensive tilt applied due to escalating Middle East geopolitical risk."
  },
  "sentiment_updates": {
    "EQQQ.L": { "rsi": 78, "bullish": 0.60, "bearish": 0.40, "label": "positive", "exhaustion_alert": true },
    "SGLN.L": { "rsi": 45, "bullish": 0.50, "bearish": 0.50, "label": "neutral", "exhaustion_alert": false }
  }
}

ALSO include these additional top-level keys in the JSON:

"summary_hints": {
  "var":    "One concise line (≤12 words). Portfolio-aware VaR advice referencing actual defensive holdings.",
  "beta":   "One concise line (≤12 words). Beta advice referencing actual tickers held.",
  "sharpe": "One concise line (≤12 words). Sharpe/efficiency advice referencing actual drag assets or quality.",
  "cagr":   "One concise line (≤12 words). Real return advice referencing the CMA-blended growth outlook."
},
"quant_intelligence": "2–3 sentences. Institutional-grade diversification commentary. Mention the specific DR value, reference 1–2 actual tickers held, and give a precise actionable insight.",
"market_outlook": [
  {"l": "Short label (2-4 words)", "v": "Short value/title (3-7 words)", "d": "2-3 sentence deep institutional analysis with specific data points, referencing actual holdings where relevant.", "i": "emoji"}
],
"strategic_commentary": "2-3 sentences. Portfolio-specific strategic observation referencing the client's Sharpe ratio, factor exposures, and allocation efficiency. Must mention actual tickers held.",
"future_outlook": "2-3 sentences. Forward-looking macro thesis with specific catalysts, rotation targets, and risk scenarios. Reference actual market conditions and the client's positioning."

market_outlook RULES:
- EXACTLY 5 items covering these topics in order: (1) Macro Regime, (2) Sector Rotation, (3) Regional Alpha, (4) Valuation Framework, (5) Strategic Hedge
- Each item MUST reference current market conditions and the client's actual holdings
- Labels (l) must be concise (2-4 words), values (v) must be punchy titles (3-7 words)
- Descriptions (d) must be 2-3 sentences of dense institutional-grade analysis
- Do NOT use generic/boilerplate text — every point must reflect TODAY's macro environment

RULES:
- severity must be one of: GEO-RISK, MACRO-SHOCK, MOMENTUM, SECTOR-ROTATE
- action must be "Buy" or "Trim"
- tactical_blueprint.target values must sum to exactly 1.0
- sentiment_updates must include every ticker in the portfolio — no omissions
- summary_hints values must NOT be generic — they must reference the client's actual holdings
- UK TAX COMPLIANCE (NON-NEGOTIABLE for GBP portfolios): Accumulating (Acc) share class \
funds held in a GIA are NOT tax-free. HMRC taxes Excess Reportable Income (ERI) from Acc \
funds as dividend income even though no cash is distributed to the investor. NEVER advise \
switching to an Acc share class to reduce or avoid dividend tax in a GIA — this is factually \
wrong and a regulatory compliance error. For GIA holdings, the correct guidance is: (1) Bed & \
ISA transfer at the start of the UK tax year (before April 5th) to shelter future gains and \
income inside an ISA wrapper, or (2) hold genuine capital-growth assets with zero distributable \
income. Only ISA and SIPP wrappers provide legal UK tax shelter.
"""

# Ticker base-name taxonomy (strip exchange suffix before matching)
_GROWTH_BASES    = {"EQQQ","AINF","DAGB","QQQ","ARKK","SMH","SOXX","MAGS","SCHG","VUG","QQQM","IGM","IIND","WTEC","LGQG","ROBO"}
_DEFENSIVE_BASES = {"SGLN","GLD","IAU","GLDM","PHAU","TLT","BND","AGG","IGLT","VGLT","IGLS","HMSO","TN28","XGSD"}
_GOLD_BASES      = {"GLD","SGLN","IAU","GLDM","PHAU","HMSO","IGLN","SGLP"}

GLOBAL_WATCHLIST = {
    "VUSA.L": {"name": "Vanguard S&P 500", "strategy": "Momentum", "type": "Equity"},
    "VUKE.L": {"name": "Vanguard FTSE 100", "strategy": "Mean Reversion", "type": "Equity"},
    "VIXL.L": {"name": "WisdomTree VIX Short-Term Futures", "strategy": "Volatility", "type": "Hedge"},
    "IGLN.L": {"name": "iShares Physical Gold", "strategy": "Volatility", "type": "Safe Haven"},
    "ISF.L":  {"name": "iShares FTSE 100", "strategy": "Arbitrage", "type": "Equity"},
    "3BRL.L": {"name": "WisdomTree S&P 500 3x Daily", "strategy": "Scalping", "type": "Leveraged"},
    "I500.L": {"name": "iShares S&P 500 Swap", "strategy": "Mean Reversion", "type": "Equity"},
}

# Known fund TERs (annual ongoing charge) for common ETFs — used when yfinance doesn't return it
_KNOWN_TER: Dict[str, float] = {
    "EQQQ":0.003,"CNDX":0.003,"QQQ":0.002,"QQQM":0.0015,
    "AINF":0.004,"IIND":0.0019,"DAGB":0.0025,"WTEC":0.004,"LGQG":0.0029,
    "ROBO":0.008,"MAGS":0.002,
    "SGLN":0.0012,"GLD":0.004,"IAU":0.0025,"GLDM":0.001,"PHAU":0.0015,
    "HMSO":0.0015,"IGLN":0.0012,"SGLP":0.0012,
    "TN28":0.0007,"IGLT":0.0007,"VGLT":0.001,"TLT":0.0015,"XGSD":0.0007,
    "BND":0.0003,"AGG":0.0003,
    "VWRP":0.0022,"VWRL":0.0022,"IWDA":0.002,"SWDA":0.002,
    "SPY":0.0009,"CSPX":0.0007,"VUAG":0.0007,"ISF":0.0007,
    "SCHD":0.0006,"WQDV":0.0025,"VIG":0.0006,"DGRW":0.0028,
    "SMH":0.0035,"SOXX":0.0035,"ARKK":0.0068,
    "XGLD":0.0012,"RBTX":0.008,
    "0P0000XW0J": 0.0092, # Invesco UK Eq High Inc
    "0P0000X63C": 0.0099, # Jupiter India (0P0000TKZO was incorrect)
}

# Ticker resolver for UK Funds/SEDOLs which yfinance can't natively lookup.
# Maps common II symbols/SEDOLs to Yahoo Finance fund tickers (0P...L format).
_FUND_TICKER_MAP = {
    "B8N46L7": "0P0000XW0J.L",  # Invesco UK Eq High Inc UK Z Acc
    "B4TZHH9": "0P0000X63C.L",  # Jupiter India I Acc (Corrected from TKZO)
    "EQQQ": "EQQQ.L",
    "VWRP": "VWRP.L",
    "VWRL": "VWRL.L",
    "VUSA": "VUSA.L",
    "VUKE": "VUKE.L",
}

def _resolve_ticker(symbol: str) -> str:
    """Helper to resolve SEDOLs or ambiguous tickers to Yahoo Finance symbols."""
    if not symbol: return symbol
    clean = symbol.upper().strip()
    # Strip composite or SEDOL prefix
    if clean.startswith("SEDOL:"):
        clean = clean[6:]
    # Try direct lookup first
    if clean in _FUND_TICKER_MAP:
        return _FUND_TICKER_MAP[clean]
    # Try without .L suffix (e.g. "B8N46L7.L" → "B8N46L7")
    bare = clean.removesuffix(".L")
    if bare != clean and bare in _FUND_TICKER_MAP:
        return _FUND_TICKER_MAP[bare]
    return symbol.upper().strip()


def _is_sedol_ticker(tk: str) -> bool:
    """Check if a ticker looks like a SEDOL code (alphanumeric, 7 chars, no '.')."""
    bare = tk.upper().strip().removesuffix(".L")
    return len(bare) == 7 and bare.isalnum() and not bare.startswith("0P")

# Fraction of each ETF's NAV denominated in non-GBP currencies.
# LSE-listed ETFs trade in GBP but often track USD-denominated indices.
# Used to compute meaningful FX sensitivity when base_currency == "GBP".
_UNDERLYING_FX: Dict[str, float] = {
    "EQQQ":0.98,"CNDX":0.98,"QQQ":0.98,"QQQM":0.98,
    "AINF":0.90,"DAGB":0.80,"WTEC":0.90,"LGQG":0.90,"ROBO":0.80,"MAGS":0.98,
    "IIND":0.98,"SMH":0.98,"SOXX":0.98,"ARKK":0.98,
    "SGLN":0.98,"GLD":0.98,"IAU":0.98,"GLDM":0.98,"PHAU":0.98,
    "HMSO":0.98,"IGLN":0.98,"SGLP":0.98,"XGLD":0.98,
    "CSPX":0.98,"SPY":0.98,"IVV":0.98,"VOO":0.98,"VTI":0.98,"VUAG":0.98,
    "VWRP":0.60,"VWRL":0.60,"IWDA":0.60,"SWDA":0.60,   # ~60% USD in world funds
    "SCHD":0.98,"VIG":0.98,"DGRW":0.98,"WQDV":0.40,
    # GBP-underlying (UK-only) — zero USD exposure
    "IGLT":0.00,"TN28":0.00,"ISF":0.00,"VMID":0.00,"XGSD":0.00,
    "BND":0.98,"AGG":0.98,"TLT":0.98,"VGLT":0.98,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background tasks
    asyncio.create_task(watcher_loop())
    asyncio.create_task(telegram_polling_loop())
    print("[Vesper] Background Watcher and Telegram Bot started.")
    yield
    _executor.shutdown(wait=False)

app = FastAPI(title="Vesper API v5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PortfolioRequest(BaseModel):
    holdings: Dict[str, float]
    risk_level: str = "Balanced"
    base_currency: str = "GBP"
    fallback_prices: Dict[str, float] = {}
    account_mapping: Dict[str, str] = {}   # e.g. {"EQQQ.L": "ISA", "SDIP.L": "GIA"}
    cash_balances: Dict[str, float] = {}   # e.g. {"ISA": 1000, "GIA": 500}
    enable_cio_llm: bool = True            # False → skip LLM, use rule engine only
    model_provider: str = "gemini"         # "claude" | "gemini"

    @field_validator("holdings")
    @classmethod
    def validate_holdings(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v: raise ValueError("Provide at least one ticker.")
        return {k.upper().strip(): float(q) for k, q in v.items()}


class TransactionRequest(BaseModel):
    csv_text: str
    account_type: str = "ISA"   # ISA | GIA | SIPP
    id_token: str = ""          # Firebase auth token (optional)


def safe_fetch(info: dict, key: str, default: Any = "N/A") -> Any:
    val = info.get(key)
    return val if val not in (None, "") else default

def _to_float(val: Any) -> Optional[float]:
    try: 
        f = float(val)
        return f if np.isfinite(f) else None
    except: return None

def _round(val: Any, decimals: int = 4) -> Any:
    f = _to_float(val)
    return round(f, decimals) if f is not None else "N/A"

# ── P&L Desk helpers ─────────────────────────────────────────────────────────

# Credit-side keywords → TRANSFER_IN (cash coming in)
_TRANSFER_IN_KEYWORDS = [
    "trf from", "subscription", "isa subscription", "bed & isa sub",
    "debit card payment", "payment via", "cash deposited",
    "pbb payment", "cashback",
]
# Debit-side keywords → TRANSFER_OUT (cash leaving or internal move)
_TRANSFER_OUT_KEYWORDS = [
    "bed & isa transfer",    # GIA→ISA Bed & ISA (debit side)
    "withdrawal",
    "transfer out",
]
_FEE_KEYWORDS = [
    "fee transfer", "total monthly fee", "management fee", "service charge",
]
# Descriptions starting with "PAYMENT" followed by reference → TRANSFER_IN
# e.g. "PAYMENT Q5724180656JMR S SHARMA"

def _tax_year(dt: datetime.date) -> str:
    """UK tax year: 6-Apr to 5-Apr.  2025-03-15 → '2024/25', 2025-04-06 → '2025/26'."""
    y, m, d = dt.year, dt.month, dt.day
    if m > 4 or (m == 4 and d >= 6):
        return f"{y}/{str((y + 1) % 100).zfill(2)}"
    return f"{y - 1}/{str(y % 100).zfill(2)}"

def _parse_gbp(raw: str, pence_to_pounds: bool = False) -> Optional[float]:
    """Parse '£1,234.56', '(£1,234.56)', '120.064p', 'n/a' → float or None.

    When pence_to_pounds=True, pence values (e.g. '120.064p') are converted to
    pounds (÷100).  Otherwise pence values are returned as-is (numeric part only).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "n/a":
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    # Detect pence suffix (e.g. "120.064p" or "443.94P")
    is_pence = s.lower().endswith("p") and not s.lower().startswith("£")
    if is_pence:
        s = s[:-1]  # strip trailing 'p'
    s = re.sub(r"[£,\s\ufeff]", "", s)
    try:
        v = float(s)
        if is_pence and pence_to_pounds:
            v /= 100.0
        if neg:
            v = -v
        return v
    except ValueError:
        return None

def _parse_ii_csv(csv_text: str, account_type: str) -> pd.DataFrame:
    """Parse an Interactive Investor CSV export into a DataFrame."""
    # Strip BOM
    clean = csv_text.replace("\ufeff", "")
    reader = csv.DictReader(io.StringIO(clean))
    rows = []
    for i, raw in enumerate(reader):
        date_str = (raw.get("Date") or "").strip()
        if not date_str:
            continue
        try:
            dt = datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
        except ValueError:
            continue
        settle_str = (raw.get("Settlement Date") or "").strip()
        try:
            settle = datetime.datetime.strptime(settle_str, "%d/%m/%Y").date()
        except ValueError:
            settle = dt

        sym = (raw.get("Symbol") or "").strip()
        sedol = (raw.get("Sedol") or "").strip()
        if sym.lower() == "n/a":
            sym = None
        if sedol.lower() == "n/a":
            sedol = None

        # Instrument key
        inst = None
        if sym:
            inst = _resolve_ticker(sym)
        elif sedol:
            inst = _resolve_ticker(sedol)
            if inst == sedol.upper():
                inst = f"SEDOL:{sedol.upper()}"
        
        # If sym was present but not resolved, inst is just sym.upper() (from _resolve_ticker fallback)
        # This is correct for normal tickers like VUSA.L or AAPL.

        qty_raw = (raw.get("Quantity") or "").strip()
        qty = None
        if qty_raw and qty_raw.lower() != "n/a":
            try:
                qty = float(qty_raw.replace(",", ""))
            except ValueError:
                qty = None

        price = _parse_gbp(raw.get("Price"), pence_to_pounds=True)
        debit = _parse_gbp(raw.get("Debit"))
        credit = _parse_gbp(raw.get("Credit"))
        balance = _parse_gbp(raw.get("Running Balance"))
        desc = (raw.get("Description") or "").strip()
        ref = (raw.get("Reference") or "").strip()

        rows.append({
            "row_idx": i,
            "date": dt,
            "settle_date": settle,
            "symbol": sym,
            "sedol": sedol,
            "instrument_key": inst,
            "qty": qty,
            "price": price,
            "description": desc,
            "reference": ref,
            "debit": debit,
            "credit": credit,
            "balance": balance,
            "account_type": account_type,
        })
    return pd.DataFrame(rows)


def _classify_row(row: dict) -> str:
    """Classify a transaction row into a type string."""
    desc_lower = (row.get("description") or "").lower()
    inst = row.get("instrument_key")
    raw_qty = row.get("qty")
    # Handle NaN from pandas
    qty = None if raw_qty is None or (isinstance(raw_qty, float) and np.isnan(raw_qty)) else raw_qty
    debit = row.get("debit") or 0
    credit = row.get("credit") or 0

    # ── Dividend / distribution income ─────────────────────────────────────
    # "Div ", "Dividend Grp", "Equalisation" in description, credit > 0, no trade qty
    if credit > 0 and (qty is None or qty == 0):
        if ("div " in desc_lower or "dividend " in desc_lower or "equalisation" in desc_lower):
            return "DIVIDEND"

    # ── Interest ───────────────────────────────────────────────────────────
    if "gross interest" in desc_lower and credit > 0:
        return "INTEREST"

    # ── Trade: must have instrument + qty > 0 ─────────────────────────────
    if inst and qty and qty > 0:
        if debit > 0:
            return "TRADE_BUY"
        if credit > 0:
            return "TRADE_SELL"

    # ── Transfer OUT (debit side) ──────────────────────────────────────────
    # Must check before TRANSFER_IN because "Bed & ISA Transfer" has debit
    # and ISA Subscription debits are internal moves out
    if debit > 0:
        for kw in _FEE_KEYWORDS:
            if kw in desc_lower:
                return "FEE"
        for kw in _TRANSFER_OUT_KEYWORDS:
            if kw in desc_lower:
                return "TRANSFER_OUT"
        # "2024/25 ISA Subscription" with DEBIT → money going OUT (from GIA to ISA)
        if "isa subscription" in desc_lower:
            return "TRANSFER_OUT"
        # "PAYMENT" with debit → money leaving (rare, e.g. "PAYMENT Q57..." debit)
        if desc_lower.startswith("payment ") and debit > 0:
            return "TRANSFER_OUT"

    # ── Transfer IN (credit side) ──────────────────────────────────────────
    if credit > 0:
        for kw in _TRANSFER_IN_KEYWORDS:
            if kw in desc_lower:
                return "TRANSFER_IN"
        # "PAYMENT Q57..." with credit → money coming in
        if desc_lower.startswith("payment "):
            return "TRANSFER_IN"

    return "OTHER"


def _run_pnl_engine(df: pd.DataFrame, account_type: str) -> dict:
    """
    Pooled Average Cost P&L engine.
    Processes a DataFrame of transactions and returns comprehensive analytics.
    """
    if df.empty:
        return {
            "account_type": account_type,
            "transaction_count": 0,
            "total_personal_contribution": 0,
            "total_book_cost": 0,
            "total_earnings": 0,
            "capital_gains": {"ytd": 0, "total": 0, "by_year": {}},
            "dividends": {"total": 0, "by_year": {}},
            "interest": {"total": 0, "by_year": {}},
            "open_positions": {},
            "yearly": {},
            "monthly": {},
            "disposals": [],
            "transfer_in_detail": [],
            "transfer_out_detail": [],
            "dividend_detail": [],
        }

    # ── 1. Classify every row ────────────────────────────────────────────────
    records = df.to_dict("records")
    for r in records:
        r["tx_type"] = _classify_row(r)

    # ── 2. Deterministic sort: date ASC → BUY=0 / OTHER=1 / SELL=2 → row_idx
    _SORT_PRIORITY = {"TRADE_BUY": 0, "TRADE_SELL": 2}
    records.sort(key=lambda r: (
        r["date"],
        _SORT_PRIORITY.get(r["tx_type"], 1),
        r["row_idx"],
    ))

    # ── 3. Process each row ──────────────────────────────────────────────────
    pools: Dict[str, Dict] = {}          # instrument → {qty, total_cost}
    last_prices: Dict[str, float] = {}   # instrument → latest per-unit price (£)
    inst_names: Dict[str, str] = {}      # instrument → human name from description
    inst_sedols: Dict[str, str] = {}     # instrument → original SEDOL code
    transfer_window_end: Optional[datetime.date] = None

    total_personal_contribution = 0.0
    cg_by_year: Dict[str, float] = {}
    div_by_year: Dict[str, float] = {}
    int_by_year: Dict[str, float] = {}
    yearly: Dict[str, Dict] = {}         # tax_year → accumulators
    monthly: Dict[str, Dict] = {}        # YYYY-MM → accumulators

    disposals: List[dict] = []
    transfer_in_detail: List[dict] = []
    transfer_out_detail: List[dict] = []
    dividend_detail: List[dict] = []

    def _ensure_year(ty: str):
        if ty not in yearly:
            yearly[ty] = {
                "realized_gains": 0, "dividends": 0, "interest": 0,
                "total_proceeds": 0, "total_allowable_cost": 0,
                "gains": 0, "losses": 0, "net_profit": 0,
            }

    def _ensure_month(mk: str):
        if mk not in monthly:
            monthly[mk] = {"realized_gains": 0, "dividends": 0, "interest": 0, "net_profit": 0}

    def _safe_num(v):
        """Convert NaN/None to 0."""
        if v is None:
            return 0
        if isinstance(v, float) and np.isnan(v):
            return 0
        return float(v)

    for r in records:
        tx = r["tx_type"]
        dt = r["date"]
        ty = _tax_year(dt)
        mk = dt.strftime("%Y-%m")
        _ensure_year(ty)
        _ensure_month(mk)
        raw_inst = r.get("instrument_key")
        inst = None if raw_inst is None or (isinstance(raw_inst, float) and np.isnan(raw_inst)) else raw_inst
        qty = _safe_num(r.get("qty"))
        debit = _safe_num(r.get("debit"))
        credit = _safe_num(r.get("credit"))
        desc = r.get("description") or ""
        date_str = dt.isoformat()

        if tx == "TRANSFER_IN":
            total_personal_contribution += credit
            transfer_window_end = dt + datetime.timedelta(days=10)
            transfer_in_detail.append({
                "date": date_str, "tax_year": ty,
                "symbol": inst or "CASH",
                "amount": round(credit, 2), "type": "Transfer In",
                "description": desc,
            })

        elif tx == "TRADE_BUY":
            cost = debit
            pool = pools.setdefault(inst, {"qty": 0, "total_cost": 0})
            pool["qty"] += qty
            pool["total_cost"] += cost
            # Track last per-unit price from CSV (for SEDOL funds Yahoo can't resolve)
            row_price = r.get("price")
            if row_price and isinstance(row_price, (int, float)) and not np.isnan(row_price) and row_price > 0:
                last_prices[inst] = float(row_price)
            elif qty > 0 and cost > 0:
                last_prices[inst] = cost / qty  # derive from total ÷ qty
            # Extract fund name from description (e.g. "Royal London Short Term Money Mkt Y Acc")
            if inst and desc and inst not in inst_names:
                inst_names[inst] = desc.split(" - ")[0].strip() if " - " in desc else desc.strip()
            # Track original SEDOL code for reverse lookup
            raw_sedol = r.get("sedol")
            if raw_sedol and isinstance(raw_sedol, str) and raw_sedol.lower() != "nan" and inst not in inst_sedols:
                inst_sedols[inst] = raw_sedol.upper().strip()

        elif tx == "TRADE_SELL":
            proceeds = credit
            pool = pools.get(inst, {"qty": 0, "total_cost": 0})

            # ── Transfer-in asset detection ──────────────────────────────
            in_window = (transfer_window_end is not None and dt <= transfer_window_end)
            if pool["qty"] < 0.001 and in_window:
                # Sell without prior buy inside transfer window
                # → treat as transferred shares arriving; gain = 0
                pool_entry = pools.setdefault(inst, {"qty": 0, "total_cost": 0})
                pool_entry["qty"] += qty       # add shares as if bought
                pool_entry["total_cost"] += proceeds  # cost basis = proceeds
                # Count inferred asset transfer as personal contribution
                total_personal_contribution += proceeds
                transfer_in_detail.append({
                    "date": date_str, "tax_year": ty, "symbol": inst,
                    "amount": round(proceeds, 2),
                    "type": "Asset Transfer In (inferred)",
                    "description": f"Sell with no prior buy within transfer window — treated as asset arrival. {desc}",
                })
                # Then immediately sell them → gain = 0
                pool_entry["qty"] -= qty
                pool_entry["total_cost"] -= proceeds
                disposals.append({
                    "date": date_str, "tax_year": ty, "symbol": inst,
                    "qty": round(qty, 4), "proceeds": round(proceeds, 2),
                    "allowable_cost": round(proceeds, 2), "gain": 0,
                })
                yearly[ty]["total_proceeds"] += proceeds
                yearly[ty]["total_allowable_cost"] += proceeds
            else:
                # Normal sell against pool
                if pool["qty"] > 0.001:
                    avg_cost = pool["total_cost"] / pool["qty"]
                else:
                    avg_cost = 0

                allowable_cost = round(avg_cost * qty, 2)
                gain = round(proceeds - allowable_cost, 2)

                # Deduct from pool
                pool["qty"] -= qty
                pool["total_cost"] -= allowable_cost
                if pool["qty"] < 0.001:
                    pool["qty"] = 0
                    pool["total_cost"] = 0

                pools[inst] = pool

                disposals.append({
                    "date": date_str, "tax_year": ty, "symbol": inst,
                    "qty": round(qty, 4), "proceeds": round(proceeds, 2),
                    "allowable_cost": round(allowable_cost, 2),
                    "gain": gain,
                })

                cg_by_year[ty] = round(cg_by_year.get(ty, 0) + gain, 2)
                yearly[ty]["realized_gains"] += gain
                yearly[ty]["total_proceeds"] += proceeds
                yearly[ty]["total_allowable_cost"] += allowable_cost
                if gain >= 0:
                    yearly[ty]["gains"] += gain
                else:
                    yearly[ty]["losses"] += gain  # negative
                
                monthly[mk]["realized_gains"] += gain

        elif tx == "DIVIDEND":
            div_by_year[ty] = round(div_by_year.get(ty, 0) + credit, 2)
            yearly[ty]["dividends"] += credit
            monthly[mk]["dividends"] += credit
            div_sym = inst if inst else (desc.split()[1] if len(desc.split()) > 1 else "Unknown")
            dividend_detail.append({
                "date": date_str, "tax_year": ty,
                "symbol": div_sym,
                "amount": round(credit, 2), "type": "Dividend",
                "description": desc,
            })

        elif tx == "INTEREST":
            int_by_year[ty] = round(int_by_year.get(ty, 0) + credit, 2)
            yearly[ty]["interest"] += credit
            monthly[mk]["interest"] += credit

        elif tx == "TRANSFER_OUT":
            total_personal_contribution -= debit
            transfer_out_detail.append({
                "date": date_str, "tax_year": ty,
                "symbol": inst or "CASH",
                "amount": round(debit, 2), "type": "Transfer Out",
                "description": desc,
            })

        elif tx == "FEE":
            # Fees are not subtracted from Total Contributed per user requirement.
            pass

    # ── 4. Build open positions ──────────────────────────────────────────────
    open_positions = {}
    total_book_cost = 0
    for inst, pool in pools.items():
        if pool["qty"] > 0.001:
            avg = round(pool["total_cost"] / pool["qty"], 4)
            tc = round(pool["total_cost"], 2)
            pos_entry = {
                "qty": round(pool["qty"], 4),
                "avg_cost": avg,
                "total_cost": tc,
            }
            # For SEDOL-keyed positions, include broker price, name & SEDOL (Yahoo can't resolve these)
            if inst.startswith("SEDOL:") or inst.startswith("0P"):
                bp = last_prices.get(inst)
                if bp and bp > 0:
                    pos_entry["broker_price"] = round(bp, 4)
                nm = inst_names.get(inst)
                if nm:
                    pos_entry["name"] = nm
                sd = inst_sedols.get(inst)
                if sd:
                    pos_entry["sedol"] = sd
            open_positions[inst] = pos_entry
            total_book_cost += tc

    # ── 5. Finalize yearly summaries ─────────────────────────────────────────
    for ty, y in yearly.items():
        y["realized_gains"] = round(y["realized_gains"], 2)
        y["dividends"] = round(y["dividends"], 2)
        y["interest"] = round(y["interest"], 2)
        y["total_proceeds"] = round(y["total_proceeds"], 2)
        y["total_allowable_cost"] = round(y["total_allowable_cost"], 2)
        y["gains"] = round(y["gains"], 2)
        y["losses"] = round(y["losses"], 2)
        y["net_profit"] = round(y["realized_gains"] + y["dividends"] + y["interest"], 2)

    # ── 5b. Finalize monthly summaries ───────────────────────────────────────
    for mk, m in monthly.items():
        m["realized_gains"] = round(m["realized_gains"], 2)
        m["dividends"] = round(m["dividends"], 2)
        m["interest"] = round(m["interest"], 2)
        m["net_profit"] = round(m["realized_gains"] + m["dividends"] + m["interest"], 2)

    # ── 6. Totals ────────────────────────────────────────────────────────────
    total_cg = round(sum(cg_by_year.values()), 2)
    total_div = round(sum(div_by_year.values()), 2)
    total_int = round(sum(int_by_year.values()), 2)
    total_earnings = round(total_cg + total_div + total_int, 2)

    cur_ty = _tax_year(datetime.date.today())

    return {
        "account_type": account_type,
        "transaction_count": len(records),
        "total_personal_contribution": round(total_personal_contribution, 2),
        "total_book_cost": round(total_book_cost, 2),
        "total_earnings": total_earnings,
        "capital_gains": {
            "ytd": round(cg_by_year.get(cur_ty, 0), 2),
            "total": total_cg,
            "by_year": cg_by_year,
        },
        "dividends": {"total": total_div, "by_year": div_by_year},
        "interest": {"total": total_int, "by_year": int_by_year},
        "open_positions": open_positions,
        "yearly": yearly,
        "monthly": monthly,
        "disposals": disposals,
        "transfer_in_detail": transfer_in_detail,
        "transfer_out_detail": transfer_out_detail,
        "dividend_detail": dividend_detail,
    }


def _aggregate_analytics(*results: dict) -> dict:
    """Merge multiple per-account P&L results into an 'all' view."""
    accounts = [r for r in results if r.get("transaction_count", 0) > 0]
    if not accounts:
        return _run_pnl_engine(pd.DataFrame(), "all")
    if len(accounts) == 1:
        merged = copy.deepcopy(accounts[0])
        merged["account_type"] = "all"
        return merged

    r2 = lambda v: round(v, 2)

    # Merge open positions
    merged_pos: Dict[str, Dict] = {}
    for a in accounts:
        for sym, p in a.get("open_positions", {}).items():
            if sym not in merged_pos:
                merged_pos[sym] = {"qty": 0, "cost": 0, "broker_price": None, "name": None, "sedol": None}
            merged_pos[sym]["qty"] += p["qty"]
            merged_pos[sym]["cost"] += p["total_cost"]
            if p.get("broker_price"):
                merged_pos[sym]["broker_price"] = p["broker_price"]
            if p.get("name"):
                merged_pos[sym]["name"] = p["name"]
            if p.get("sedol"):
                merged_pos[sym]["sedol"] = p["sedol"]

    open_positions = {}
    for sym, p in merged_pos.items():
        if p["qty"] > 0.001:
            entry = {
                "qty": round(p["qty"], 4),
                "avg_cost": round(p["cost"] / p["qty"], 4),
                "total_cost": round(p["cost"], 2),
            }
            if p.get("broker_price"):
                entry["broker_price"] = p["broker_price"]
            if p.get("name"):
                entry["name"] = p["name"]
            if p.get("sedol"):
                entry["sedol"] = p["sedol"]
            open_positions[sym] = entry

    total_book_cost = sum(p["total_cost"] for p in open_positions.values())

    # Sum scalars
    total_contrib = sum(a.get("total_personal_contribution", 0) for a in accounts)
    total_earnings = sum(a.get("total_earnings", 0) for a in accounts)
    tx_count = sum(a.get("transaction_count", 0) for a in accounts)

    # Merge by-year maps
    def merge_by_year(*dicts):
        out = {}
        for d in dicts:
            for yr, v in (d or {}).items():
                out[yr] = round(out.get(yr, 0) + v, 2)
        return out

    all_cg = merge_by_year(*[a["capital_gains"]["by_year"] for a in accounts])
    all_dv = merge_by_year(*[a["dividends"]["by_year"] for a in accounts])
    all_in = merge_by_year(*[a["interest"]["by_year"] for a in accounts])

    cur_ty = _tax_year(datetime.date.today())

    # Merge yearly summaries
    all_years = set()
    for a in accounts:
        all_years.update(a.get("yearly", {}).keys())
    sum_fields = ["realized_gains", "dividends", "interest", "total_proceeds",
                  "total_allowable_cost", "gains", "losses"]
    yearly = {}
    for yr in sorted(all_years):
        merged_yr = {}
        for f in sum_fields:
            merged_yr[f] = r2(sum(a.get("yearly", {}).get(yr, {}).get(f, 0) for a in accounts))
        merged_yr["net_profit"] = r2(merged_yr["realized_gains"] + merged_yr["dividends"] + merged_yr["interest"])
        yearly[yr] = merged_yr

    # Merge monthly summaries
    all_months = set()
    for a in accounts:
        all_months.update(a.get("monthly", {}).keys())
    monthly = {}
    for mk in sorted(all_months):
        m_rg = sum(a.get("monthly", {}).get(mk, {}).get("realized_gains", 0) for a in accounts)
        m_dv = sum(a.get("monthly", {}).get(mk, {}).get("dividends", 0) for a in accounts)
        m_in = sum(a.get("monthly", {}).get(mk, {}).get("interest", 0) for a in accounts)
        monthly[mk] = {
            "realized_gains": r2(m_rg),
            "dividends": r2(m_dv),
            "interest": r2(m_in),
            "net_profit": r2(m_rg + m_dv + m_in),
        }

    # Concat lists
    disposals = sorted(
        [d for a in accounts for d in a.get("disposals", [])],
        key=lambda x: x["date"])
    transfer_in_detail = sorted(
        [d for a in accounts for d in a.get("transfer_in_detail", [])],
        key=lambda x: x["date"])
    transfer_out_detail = sorted(
        [d for a in accounts for d in a.get("transfer_out_detail", [])],
        key=lambda x: x["date"])
    dividend_detail = sorted(
        [d for a in accounts for d in a.get("dividend_detail", [])],
        key=lambda x: x["date"])

    return {
        "account_type": "all",
        "transaction_count": tx_count,
        "total_personal_contribution": r2(total_contrib),
        "total_book_cost": r2(total_book_cost),
        "total_earnings": r2(total_earnings),
        "capital_gains": {
            "ytd": r2(all_cg.get(cur_ty, 0)),
            "total": r2(sum(all_cg.values())),
            "by_year": all_cg,
        },
        "dividends": {"total": r2(sum(all_dv.values())), "by_year": all_dv},
        "interest": {"total": r2(sum(all_in.values())), "by_year": all_in},
        "open_positions": open_positions,
        "yearly": yearly,
        "monthly": monthly,
        "disposals": disposals,
        "transfer_in_detail": transfer_in_detail,
        "transfer_out_detail": transfer_out_detail,
        "dividend_detail": dividend_detail,
    }


# ── Firestore persistence for P&L Desk ──────────────────────────────────────

def _save_transactions_to_firestore(uid: str, csv_text: str, account_type: str) -> Tuple[int, int]:
    """Save parsed CSV rows to Firestore, deduplicating by reference field.
    Returns (new_count, dup_count)."""
    if not _FIREBASE_OK or not _FS_CLIENT:
        return 0, 0
    df = _parse_ii_csv(csv_text, account_type)
    coll = _FS_CLIENT.collection("users").document(uid).collection("transactions")
    new_count = 0
    dup_count = 0
    for _, row in df.iterrows():
        ref = row.get("reference") or ""
        if not ref:
            continue
        doc_id = f"{account_type}_{ref}"
        doc_ref = coll.document(doc_id)
        if doc_ref.get().exists:
            dup_count += 1
            continue
        doc_ref.set({
            "date": row["date"].isoformat(),
            "settle_date": row["settle_date"].isoformat(),
            "symbol": row.get("symbol") or "",
            "sedol": row.get("sedol") or "",
            "instrument_key": row.get("instrument_key") or "",
            "qty": row.get("qty"),
            "price": row.get("price"),
            "description": row.get("description") or "",
            "reference": ref,
            "debit": row.get("debit"),
            "credit": row.get("credit"),
            "balance": row.get("balance"),
            "account_type": account_type,
        })
        new_count += 1
    return new_count, dup_count


def _load_user_transactions(uid: str) -> dict:
    """Load all saved transactions from Firestore, run P&L engine, return aggregated."""
    if not _FIREBASE_OK or not _FS_CLIENT:
        return _run_pnl_engine(pd.DataFrame(), "all")
    coll = _FS_CLIENT.collection("users").document(uid).collection("transactions")
    docs = coll.stream()
    rows_by_acct: Dict[str, List[dict]] = {}
    for doc in docs:
        d = doc.to_dict()
        acct = d.get("account_type", "ISA")
        rows_by_acct.setdefault(acct, []).append(d)

    results = {}
    for acct, rows in rows_by_acct.items():
        # Reconstruct DataFrame
        for r in rows:
            r["date"] = datetime.datetime.strptime(r["date"], "%Y-%m-%d").date()
            r["settle_date"] = datetime.datetime.strptime(r["settle_date"], "%Y-%m-%d").date()
            r["row_idx"] = 0
        df = pd.DataFrame(rows)
        df["row_idx"] = range(len(df))
        results[acct] = _run_pnl_engine(df, acct)

    all_results = list(results.values())
    if all_results:
        results["all"] = _aggregate_analytics(*all_results)
    else:
        results["all"] = _run_pnl_engine(pd.DataFrame(), "all")

    return {"by_account": results}


async def _in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)

async def _get_fx_rate(from_curr: str, to_curr: str) -> float:
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        t = yf.Ticker(pair)
        info = await _in_thread(lambda: t.info)
        rate = info.get("regularMarketPrice") or info.get("previousClose")
        return float(rate) if rate else 1.0
    except: return 1.0

def _normalise_gbp_series(df: pd.DataFrame) -> pd.DataFrame:
    """Fix GBp/GBP unit flips in London-listed ticker OHLCV data.

    Yahoo Finance occasionally reports some rows in pence and others in pounds
    for the same .L ticker within a single time-series.  This creates 100x
    jumps/drops that corrupt RSI, ATR, and 30-day performance calculations.

    Strategy: use the **median** close as the reference scale.  Any row whose
    Close deviates by > 50× from the median is assumed to be in the wrong unit
    (pence vs pounds) and is divided or multiplied by 100 accordingly.
    """
    if df.empty or "Close" not in df.columns or len(df) < 10:
        return df
    close = df["Close"]
    median_c = close.median()
    if median_c <= 0:
        return df
    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    df = df.copy()
    # Rows where Close is ~100x the median → in pence when median is in pounds
    too_high = close > median_c * 50
    # Rows where Close is ~1/100 the median → in pounds when median is in pence
    too_low = close < median_c / 50
    if too_high.any():
        for col in price_cols:
            df.loc[too_high, col] = df.loc[too_high, col] / 100.0
    if too_low.any():
        for col in price_cols:
            df.loc[too_low, col] = df.loc[too_low, col] * 100.0
    return df

async def _fetch_full_history(tickers: List[str]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    closes = {}
    full_data = {}
    async def fetch_one(ticker: str):
        try:
            api_tk = _resolve_ticker(ticker)
            t = yf.Ticker(api_tk)
            h = await _in_thread(lambda: t.history(period="1y", auto_adjust=True))
            if h is not None and not h.empty:
                if h.index.tz is not None: h.index = h.index.tz_localize(None)
                # Fix GBp/GBP unit flips for London-listed tickers
                if ticker.endswith(".L"):
                    h = _normalise_gbp_series(h)
                return ticker, h
        except Exception as e:
            print(f"[Vesper] History fetch failed for {ticker}: {e}")
        return ticker, pd.DataFrame()
    results = await asyncio.gather(*[fetch_one(tk) for tk in tickers])
    for tk, df in results:
        if not df.empty:
            closes[tk] = df["Close"]
            full_data[tk] = df
    return pd.DataFrame(closes), full_data

def _compute_risk_history_from_prices(prices: pd.DataFrame, tickers: List[str], weights_dict: Dict[str, float]) -> Dict[str, Any]:
    empty = {
        "asset_risk": {t: {"cagr_inception": 0, "volatility": 0.2} for t in tickers}, 
        "portfolio_risk": {"cagr_inception": 0, "volatility": 0.15, "sharpe_ratio": 0, "var_95_daily": 0}, 
        "future_value": {"p10": [1.0]*11, "p50": [1.0]*11, "p90": [1.0]*11}, 
        "history": {"__portfolio__": {"dates": [], "values": []}}, 
        "risk_contributions": {t: 0 for t in tickers}, 
        "asset_vols": {t: 0.2 for t in tickers}, 
        "asset_betas": {t: 1.0 for t in tickers},
        "cma_mu": 0.07,
    }
    try:
        if prices.empty: return empty
        asset_risk, history, asset_vols = {}, {}, {}
        
        def calc_cagr(s, y):
            if y <= 0 or len(s) < 2: return 0
            try:
                if s.iloc[0] <= 0: return 0
                val = (s.iloc[-1] / s.iloc[0])**(1/y)-1
                return round(float(val), 4) if np.isfinite(val) else 0
            except: return 0

        for t in tickers:
            api_t = t.split('_')[0]
            if api_t not in prices.columns: 
                asset_risk[t] = {"cagr_inception": 0, "volatility": 0.2}
                asset_vols[t] = 0.2
                continue
            p = prices[api_t].dropna()
            if len(p) < 2: 
                asset_risk[t] = {"cagr_inception": 0, "volatility": 0.2}
                asset_vols[t] = 0.2
                continue
            last = p.index[-1]
            p_1y = p[p.index >= (last - pd.Timedelta(days=365.25))]
            if not p_1y.empty:
                normed = (p_1y / p_1y.iloc[0] * 100).round(2)
                history[t] = {"dates": [d.strftime("%Y-%m-%d") for d in normed.index[::5]], "values": normed.values[::5].tolist()}
            p_1y_rets = p_1y.pct_change().dropna().clip(-0.20, 0.20)
            vol = round(float(p_1y_rets.std()*np.sqrt(252)), 4) if len(p_1y_rets) > 5 else 0.20
            asset_vols[t] = vol
            years_total = max((p.index[-1]-p.index[0]).days/365.25, 0.1)
            asset_risk[t] = {"cagr_inception": calc_cagr(p, years_total), "volatility": vol}

        filled = prices.ffill().dropna()
        # Map composite tickers to their base ticker price column
        valid = [t for t in tickers if t.split('_')[0] in filled.columns]
        if not valid: return empty
        w = np.array([weights_dict.get(t, 0.0) for t in valid])
        if w.sum() > 0: w = w/w.sum() 
        else: w = np.ones(len(valid))/len(valid)
        
        # Build returns for composite tickers (they share price columns)
        rets_dict = {t: filled[t.split('_')[0]].pct_change().dropna().clip(-0.20, 0.20) for t in valid}
        rets = pd.DataFrame(rets_dict)
        if rets.empty: return empty

        cov_matrix = rets.cov() * 252
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        port_vol = np.sqrt(max(port_var, 1e-9))
        # Sanity cap: >100% annualised vol is a sign of bad data even after clipping.
        # Fall back to weighted-average individual vols (more stable).
        if port_vol > 1.0:
            port_vol = min(sum(w[i] * asset_vols.get(valid[i], 0.15) for i in range(len(valid))), 0.60)
        # MCTR: Marginal Contribution to Total Risk
        # MCTR_i = w_i × (Σ @ w)[i] / port_vol  — covariance-aware, so high-beta growth
        # assets dominate the risk budget while low-corr defensives contribute minimally.
        cov_vals = cov_matrix.values                       # annualised (n × n) array
        marginal  = cov_vals @ w                           # (n,) — marginal variance per asset
        mctr_raw  = w * marginal / max(port_vol, 1e-9)    # (n,) — unnormalised MCTR
        mctr_sum  = mctr_raw.sum()
        risk_contributions = {
            tk: round(float(mctr_raw[i] / max(mctr_sum, 1e-9)), 4)
            for i, tk in enumerate(valid)
        }
        
        asset_betas = {}
        if "SPY" in filled.columns:
            spy_rets = filled["SPY"].pct_change().dropna()
            for tk in valid:
                try:
                    common_idx = rets[tk].index.intersection(spy_rets.index)
                    if len(common_idx) > 10:
                        c = np.cov(rets[tk].loc[common_idx], spy_rets.loc[common_idx])[0,1]
                        v = np.var(spy_rets.loc[common_idx])
                        asset_betas[tk] = round(float(c/v), 3) if v != 0 else 1.0
                    else: asset_betas[tk] = 1.0
                except: asset_betas[tk] = 1.0
        else:
            asset_betas = {t: 1.0 for t in tickers}

        port_rets = rets.values @ w
        port_series = pd.Series(port_rets, index=rets.index)
        p_1y_cum = (1 + port_series[port_series.index >= (filled.index[-1] - pd.Timedelta(days=365.25))]).cumprod()
        history["__portfolio__"] = {"dates": [d.strftime("%Y-%m-%d") for d in p_1y_cum.index[::5]], "values": (p_1y_cum[::5] * 100).round(2).tolist()}

        # Annualised portfolio CAGR from winsorized cumulative returns.
        # When shared history is < 1 year (new ETF limits joint data),
        # fall back to the weighted average of per-asset CAGRs — far more stable.
        port_cum = (1 + port_series).cumprod()
        y_p = max(len(port_series) / 252, 0.1)
        weighted_asset_cagr = float(sum(w[i] * asset_risk.get(valid[i], {}).get("cagr_inception", 0.0) for i in range(len(valid))))
        if y_p >= 1.0 and len(port_cum) > 0 and port_cum.iloc[-1] > 0:
            raw_cagr = float(port_cum.iloc[-1] ** (1 / y_p) - 1)
            # Sanity: any annualised CAGR outside [-50%, +100%] means bad data survived clipping
            cagr_p = round(raw_cagr if -0.50 <= raw_cagr <= 1.00 else weighted_asset_cagr, 4)
        else:
            # Short joint history → per-asset weighted average is more reliable
            cagr_p = round(weighted_asset_cagr, 4)
        portfolio_risk = {"cagr_inception": cagr_p, "volatility": round(float(port_vol), 4), "sharpe_ratio": round((cagr_p-RISK_FREE_RATE)/port_vol, 2) if port_vol > 0 else 0, "var_95_daily": round(float(1.645 * (port_vol / np.sqrt(252))), 4)}
        
        # CMA GBM Monte Carlo — forward-neutral Capital Market Assumptions per asset class.
        # Replaces bootstrap resampling which was biased by the recent bull-market period.
        _CMA = {"growth": (0.085, 0.22), "core": (0.070, 0.16), "defensive": (0.040, 0.12)}
        def _cma_bucket(tk: str) -> str:
            base = tk.split('.')[0].upper()
            if base in _GROWTH_BASES:    return "growth"
            if base in _DEFENSIVE_BASES: return "defensive"
            return "core"
        mu_blend  = float(sum(w[i] * _CMA[_cma_bucket(valid[i])][0] for i in range(len(valid))))
        # Arithmetic weighted-average vol (assumes full correlation — conservative upper bound).
        sig_blend = float(sum(w[i] * _CMA[_cma_bucket(valid[i])][1] for i in range(len(valid))))
        sig_blend = max(sig_blend, 0.10)   # floor: even a cash-heavy portfolio has residual vol
        n_sims, n_years = 2000, 10
        rng = np.random.default_rng(42)
        # Annual log-return ~ N((μ − 0.5σ²), σ²) — exact GBM, one draw per year per sim.
        annual_log_drift = mu_blend - 0.5 * sig_blend ** 2
        log_rets_sim = rng.normal(annual_log_drift, sig_blend, (n_sims, n_years))
        cum_log = np.cumsum(log_rets_sim, axis=1)          # (n_sims, 10)
        paths   = np.column_stack([np.ones(n_sims), np.exp(cum_log)])  # (n_sims, 11) Year 0–10
        cma_mu  = mu_blend   # store for real_cagr downstream
        future_value = {
            "median_cagr": round(float(np.percentile(paths[:, -1], 50) ** (1 / n_years) - 1), 4),
            "p10": [round(x, 4) for x in np.percentile(paths, 10, axis=0).tolist()],
            "p50": [round(x, 4) for x in np.percentile(paths, 50, axis=0).tolist()],
            "p90": [round(x, 4) for x in np.percentile(paths, 90, axis=0).tolist()],
        }
        
        return {"asset_risk": asset_risk, "portfolio_risk": portfolio_risk, "future_value": future_value, "history": history, "risk_contributions": risk_contributions, "asset_vols": asset_vols, "asset_betas": asset_betas, "cma_mu": cma_mu}
    except:
        traceback.print_exc()
        return empty

def _compute_rsi(series: pd.Series, window: int = 14) -> float:
    if len(series) < window + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = -1 * delta.clip(upper=0).rolling(window=window, min_periods=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def _compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    if len(df) < window + 1: return 0.0
    high = df['High'] if 'High' in df.columns else df['Close']
    low = df['Low'] if 'Low' in df.columns else df['Close']
    close = df['Close']
    tr = pd.concat([high - low, 
                    (high - close.shift(1)).abs(), 
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else 0.0

def _get_advisor_logic(symbol: str, hist: pd.DataFrame, sentiment_val: float, a: dict, vix: float, benchmark_hist: pd.DataFrame, cycle: str, alpha_signal: dict = None, insider_signal: dict = None) -> dict:
    if hist.empty or len(hist) < 14:
        return {
            "score": 50, "strat": "Neutral", "action": "HOLD", 
            "action_instruction": "HOLD (AWAIT DATA)",
            "why": "Insufficient data", "detailed_advisor_report": "Insufficient data.",
            "probability": "50%", "strategy_narrative": "None", "evidence": {},
            "exhaustion_status": "Neutral"
        }

    price = hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[-2]
    ma50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else price
    ma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else ma50
    rsi = _compute_rsi(hist['Close'])
    atr = _compute_atr(hist)
    day_gain = (price / prev_close - 1) * 100
    atr_pct = (atr / price) * 100 if price > 0 else 0
    
    # Data-driven gain/loss proxy: 30-day performance
    perf_30d = (price / hist['Close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0.0
    
    base_tk = symbol.split('_')[0].split('.')[0].upper()
    is_watchlist = symbol in GLOBAL_WATCHLIST
    
    score = 50
    strat = "Neutral"
    action = "HOLD"
    probability = 50
    
    # 1. Momentum: Strong trend and healthy RSI
    is_momentum = price > ma50 and 55 <= rsi <= 70
    if is_momentum:
        score += 30; strat = "Momentum"; action = "BUY (MOMENTUM RIDE)"; probability = 70

    # 2. Mean Reversion: Oversold or significant recent drop
    is_reversion = rsi < 35 or perf_30d < -10
    if is_reversion:
        score += 40; strat = "Mean Reversion"; action = "BUY (REVERSION BUY)"; probability = 80

    # 3. Pairs Trading: Correlation break with global benchmark
    if not benchmark_hist.empty and len(hist) >= 60:
        common_idx = hist.index.intersection(benchmark_hist.index)
        if len(common_idx) > 30:
            corr = hist['Close'].loc[common_idx].corr(benchmark_hist['Close'].loc[common_idx])
            if corr < 0.55:
                score += 25; strat = "Pairs Trading"; action = "BUY (REL-VAL OPPORTUNITY)"; probability = 65

    # 4. Volatility Defense: Flight to safety during high VIX
    is_vol_defense = base_tk in _GOLD_BASES or "CMOP" in base_tk or "VIX" in base_tk
    if vix > 25 and is_vol_defense:
        score = max(score, 85); strat = "Volatility Defense"; action = "HEDGE (VOL DEFENSE)"; probability = 95

    # 5. Scalping: High volatility and intraday/daily gain
    if atr_pct > 2.5 and abs(day_gain) > 1.5:
        score = max(score, 80); strat = "Scalping"; action = "TRIM (QUICK SCALP)" if day_gain > 0 else "BUY (SCALP DIP)"; probability = 85

    # 6. Arbitrage: placeholder — requires real NAV vs market price data to implement
    # (Removed hardcoded ISF +15 bias — was unconditional fake signal)

    # 7. Hedging: Negative sentiment and breaking technicals
    is_hedge_candidate = (sentiment_val < -0.3 and price < ma50)
    if is_hedge_candidate:
        score = max(score, 85); strat = "Hedging"; action = "HEDGE (SENTIMENT HEDGE)"; probability = 90

    # 8. Insider Alpha (Task 1)
    is_insider_bullish = False
    if insider_signal and insider_signal.get("detected"):
        score += 20
        is_insider_bullish = True
        if strat == "Neutral": strat = "Smart Money"
        
    # 9. Lead-Lag Alpha (Task 2)
    is_delayed_breakout = False
    if alpha_signal and alpha_signal.get("status") == "Delayed Breakout Opportunity":
        score += 15
        is_delayed_breakout = True
        if strat == "Neutral": strat = "Lead-Lag"

    # Exhaustion Engine
    exhaustion_status = "Neutral"
    logic_tag = "Stable"
    if rsi > 75 and day_gain > 2: 
        exhaustion_status = "Blow-off Top"
        logic_tag = "Exit Peak"
    elif rsi > 70 and sentiment_val < 0: 
        exhaustion_status = "Bull Trap"
        logic_tag = "False Breakout"
    elif rsi < 30 and perf_30d < -15: 
        exhaustion_status = "Deep Exhaustion"
        logic_tag = "Rubber Band"
    elif rsi < 35: 
        exhaustion_status = "Oversold"
        logic_tag = "Mean Reversion"
    elif rsi > 65: 
        exhaustion_status = "Overextended"
        logic_tag = "Profit Taking"
    
    if strat == "Volatility Defense": logic_tag = "Safe Haven"
    elif strat == "Hedging": logic_tag = "Beta Hedge"
    elif is_momentum: logic_tag = "Trend Ride"
    elif is_insider_bullish: logic_tag = "Insider Accumulation"
    elif is_delayed_breakout: logic_tag = "Lagged Confirmation"

    # Regime-Based Probability Weighting (Contraction focus)
    if cycle == "Contraction":
        if strat == "Momentum": 
            score -= 30; probability -= 20
        if strat in ["Volatility Defense", "Hedging", "Mean Reversion"]: 
            score += 15; probability += 10
        # Boost safe havens in high fear
        if is_vol_defense:
            probability = max(probability, 92)

    final_score = min(100, max(0, int(score)))
    final_prob = min(99, max(5, int(probability)))
    
    # Executive Reasoning (Redundancy-Free)
    reasoning = f"{strat} alignment verified. "
    if is_insider_bullish:
        reasoning += f"Significant insider buying (£{insider_signal['amount']/1e6:.1f}M) detected. "
    if is_delayed_breakout:
        reasoning += f"Lead-lag divergence with {alpha_signal['lead_symbol']} suggests a delayed breakout opportunity. "

    if is_reversion:
        reasoning += f"Extreme price-to-mean stretch ({perf_30d:+.1f}%) suggests high-probability snap-back. Current fear is masking value."
    elif is_momentum:
        reasoning += f"Trend persistence is holding, but {cycle} regime requires aggressive stop-loss management."
    elif is_vol_defense:
        reasoning += f"VIX {vix:.2f} regime mandates flight to quality. {base_tk} acts as portfolio insurance while risk assets liquidate."
    elif is_hedge_candidate:
        reasoning += f"Sentiment/Price divergence detected. Market is ignoring deteriorating internals; protect capital now."
    else:
        reasoning += f"Macro regime ({cycle}) and technical signals confirm current posture."

    # Watchlist Comparative Advantage
    comp_adv = ""
    if is_watchlist:
        w_info = GLOBAL_WATCHLIST.get(symbol, {})
        w_type = w_info.get("type", "Asset")
        comp_adv = f"Superior {w_type} instrument for {strat} in {cycle} regimes."

    # Action Instruction
    if final_score > 80:
        if strat == "Mean Reversion": act_instr = f"BUY 5% NOW (£{price:.2f})"
        elif strat == "Volatility Defense": act_instr = "HEDGE UP (Safe Haven Rotation)"
        elif strat == "Scalping": act_instr = "TRIM 10% (Volatility Harvest)"
        elif strat == "Hedging": act_instr = "HEDGE 5% NOW (Neutralize Beta)"
        else: act_instr = f"{action} NOW"
    elif final_score < 40:
        act_instr = "TRIM (Raise Dry Powder)"
    else:
        act_instr = "HOLD (Await Signal)"

    return {
        "score": final_score,
        "strat": strat,
        "action": action,
        "probability": f"{final_prob}%",
        "sentiment_impact": "Warning" if sentiment_val < 0 else "Confirmation",
        "exhaustion_status": exhaustion_status,
        "logic_tag": logic_tag,
        "reasoning": reasoning,
        "comparative_advantage": comp_adv,
        "action_instruction": act_instr,
        "why": reasoning,
        "detailed_advisor_report": reasoning,
        "evidence": {
            "rsi": round(rsi, 1),
            "atr_pct": round(atr_pct, 2),
            "ma50": round(ma50, 2),
            "perf_30d": f"{perf_30d:+.1f}%",
            "vix": round(vix, 2),
            "sentiment": round(sentiment_val, 2),
            "exhaustion": exhaustion_status,
            "insider_bullish": is_insider_bullish,
            "delayed_breakout": is_delayed_breakout
        }
    }

def _run_apex_advisor(assets, prices, ohlc_data, p_summary, live_vix: float = None):
    # Prefer the live quote VIX passed from the main analysis function
    if live_vix is not None and live_vix > 0:
        vix_latest = live_vix
    elif "^VIX" in prices.columns and not prices["^VIX"].dropna().empty:
        vix_latest = float(prices["^VIX"].dropna().iloc[-1])
    else:
        vix_latest = 15.0
    
    vusa_p = prices["VUSA.L"].iloc[-1] if "VUSA.L" in prices.columns else 0
    vusa_ma200 = prices["VUSA.L"].rolling(200).mean().iloc[-1] if "VUSA.L" in prices.columns and len(prices) >= 200 else vusa_p
    
    # Macro Regime Detection
    cycle_indicator = "Expansion"
    if vix_latest > 25:
        cycle_indicator = "Contraction"
    elif vusa_p < vusa_ma200:
        cycle_indicator = "Trough"
    elif vusa_p > vusa_ma200 * 1.1:
        cycle_indicator = "Peak"
    
    benchmark_hist = prices["VWRL.L"] if "VWRL.L" in prices.columns else pd.Series()
    
    # Analyze Portfolio
    apex_scores = {}
    act_now = []
    
    for tk, a in assets.items():
        api_tk = tk.split('_')[0]
        hist_df = ohlc_data.get(api_tk, pd.DataFrame())
        
        s = a.get("sentiment", {})
        bull = _to_float(s.get("bullish")) if s.get("bullish") is not None else 0.5
        bear = _to_float(s.get("bearish")) if s.get("bearish") is not None else 0.5
        sentiment_val = (bull or 0.5) - (bear or 0.5)
        sentiment_is_mock = bool(s.get("_mock"))

        alpha_s = a.get("alpha_signal")
        insider_s = a.get("insider_signal")

        logic = _get_advisor_logic(tk, hist_df, sentiment_val, a, vix_latest, pd.DataFrame({"Close": benchmark_hist}), cycle_indicator, alpha_signal=alpha_s, insider_signal=insider_s)
        logic["evidence"]["sentiment_unavailable"] = sentiment_is_mock
        apex_scores[tk] = logic
        
        if logic["score"] > 80:
            act_now.append({"ticker": tk, "score": logic["score"], "action": logic["action"], "strat": logic["strat"]})

    # Analyze Watchlist
    watchlist_analysis = {}
    for tk, info in GLOBAL_WATCHLIST.items():
        hist_df = ohlc_data.get(tk, pd.DataFrame())
        # For watchlist, we could also compute alpha/insider signals if needed, but for now we pass None or dummy
        logic = _get_advisor_logic(tk, hist_df, 0.0, {}, vix_latest, pd.DataFrame({"Close": benchmark_hist}), cycle_indicator)
        watchlist_analysis[tk] = logic

    act_now = sorted(act_now, key=lambda x: x["score"], reverse=True)[:3]
    
    return {
        "cycle_indicator": cycle_indicator,
        "vix": round(vix_latest, 2),
        "act_now": act_now,
        "scores": apex_scores,
        "watchlist": watchlist_analysis
    }

def _get_advanced_intelligence(assets, portfolio, risk_level, risk_data, base_currency, rsi_cache=None):
    total_val = portfolio.get("total_value", 0)
    beta = portfolio.get("portfolio_beta", 1.0)
    var_95 = portfolio.get("var_95_daily", 0)
    asset_vols = risk_data.get("asset_vols", {})
    
    kill_switch = "SYSTEM OVERLOAD: Guardrails breached. Rotate to GLD/VIG." if (var_95 > 0.035 or beta > 1.3) else None
    
    weighted_vol_sum = sum(assets[tk]["weight"] * asset_vols.get(tk, 0.2) for tk in assets)
    div_ratio = round(weighted_vol_sum / portfolio.get("volatility", 0.15), 3) if portfolio.get("volatility", 0) > 0 else 1.0

    # FX exposure: LSE ETFs report trading currency = GBP even when underlying is USD-heavy.
    # Use _UNDERLYING_FX lookup to estimate the real non-base-currency fraction.
    if base_currency == "GBP":
        non_base_frac = sum(
            a["weight"] * _UNDERLYING_FX.get(tk.replace('_', '.').split('.')[0].upper(),
                0.0 if a.get("currency") == base_currency else 1.0)
            for tk, a in assets.items()
        )
        non_base_val = total_val * non_base_frac
    else:
        non_base_val = sum(a["value"] for a in assets.values() if a.get("currency") != base_currency)
    # Use CMA-blended forward return (not historical CAGR) for real_cagr so it's
    # cycle-neutral. Subtract 2.5% UK CPI assumption to give real purchasing-power return.
    real_cagr = round(risk_data.get("cma_mu", portfolio.get("cagr_inception", 0.075)) - 0.025, 4)

    targets = {"Conservative": 0.20, "Balanced": 0.40, "Aggressive": 0.70}
    target_g_w = targets.get(risk_level, 0.40)
    # Segment Identification — ticker-base matching takes priority over sector/PE heuristics
    growth_tks = [tk for tk, a in assets.items() if
        tk.replace('_', '.').split('.')[0].upper() in _GROWTH_BASES or
        a["sector"] in ("Technology", "Communication Services") or
        (_to_float(a.get("pe_ratio")) or 0) > 30]
    defensive_tks = [tk for tk, a in assets.items() if
        tk.replace('_', '.').split('.')[0].upper() in _DEFENSIVE_BASES or
        "Treasury" in a["name"] or "Gold" in a["name"] or "Bond" in a["name"] or
        tk.replace('_', '.').split('.')[0].upper() in ("TN28",)]
    core_tks = [tk for tk in assets if tk not in growth_tks and tk not in defensive_tks]
    
    targets = {"Conservative": 0.20, "Balanced": 0.40, "Aggressive": 0.70}
    target_g_w = targets.get(risk_level, 0.40)
    curr_g_w = sum(assets[tk]["weight"] for tk in growth_tks)
    drift = curr_g_w - target_g_w
    
    trade_suggestion = None
    cost = 0.0
    if abs(drift) > 0.05:
        # Physical Capping Logic: Ensure we never suggest selling more than 100% of an asset
        amt = abs(drift) * total_val
        cost = round(amt * 0.006, 2)
        
        if drift > 0: # Overweight Growth -> Sell Growth, Buy Core or Defensive
            sell_tk = max(growth_tks, key=lambda x: assets[x]["weight"]) if growth_tks else "Growth Segment"
            buy_tk = max(core_tks, key=lambda x: assets[x]["weight"]) if core_tks else "Core Segment"
            # Cap the trade at the actual value of the sell candidate
            actual_trade_amt = min(amt, assets.get(sell_tk, {"value": amt})["value"])
            trade_suggestion = f"Sell {base_currency} {actual_trade_amt:,.0f} of {sell_tk} and reallocate to {buy_tk}."
        else: # Underweight Growth -> Sell Core (NEVER Sell Defensive/Hedge)
            # Only sell from Core, preserve Hedges/Gilts
            sell_source = core_tks if core_tks else [tk for tk in assets if tk not in growth_tks]
            sell_tk = max(sell_source, key=lambda x: assets[x]["weight"]) if sell_source else "Portfolio"
            buy_tk = max(growth_tks, key=lambda x: assets[x]["weight"]) if growth_tks else "Growth Proxy"
            actual_trade_amt = min(amt, assets.get(sell_tk, {"value": amt})["value"])
            trade_suggestion = f"Reduce {sell_tk} by {base_currency} {actual_trade_amt:,.0f} to fund {buy_tk} expansion."

    # Multi-factor quality/safety score (0–100) per ticker.
    # The old formula bottomed out at a constant 76 for ETFs because it relied solely on
    # pe_ratio (always null for ETFs → defaults to 20) and dividend_yield (usually 0).
    # New approach uses RSI momentum deviation, beta, TER, dividend yield, and PE only
    # when actually available — producing varied, realistic scores across the portfolio.
    div_safety = {}
    for _tk, _a in assets.items():
        _rsi  = (rsi_cache or {}).get(_tk) or 50.0                          # real RSI when available; 50 = neutral fallback
        _beta = _to_float(_a.get("beta")) or 1.0
        _ter  = (_to_float(_a.get("ter"))
                 or _KNOWN_TER.get(_tk.replace('_', '.').split('.')[0].upper(), 0.003))
        _yld  = _to_float(_a.get("dividend_yield")) or 0.0
        _pe   = _to_float(_a.get("pe_ratio"))                          # None for most ETFs

        _rsi_pen  = abs(_rsi - 50) * 0.5          # 0 (RSI=50) → 15 (RSI=20 or 80)
        _beta_pen = max(0.0, (_beta - 0.4) * 11)  # 0 at β≤0.4, 17.6 at β=2.0
        _ter_pen  = _ter * 700                     # ≈2 for 0.3% TER; ≈7 for 1% TER
        _pe_pen   = (_pe * 0.8) if _pe else 0.0   # only penalises when real PE exists
        _yld_bon  = min(8.0, _yld * 200)           # up to +8 for ≥4% yield

        div_safety[_tk] = max(20, min(95, round(
            100 - _rsi_pen - _beta_pen - _ter_pen - _pe_pen + _yld_bon
        )))
    
    # Ideal Portfolio Blueprint
    ideal_mapping = {
        "Conservative": {"Growth": 0.20, "Core": 0.50, "Defensive": 0.30},
        "Balanced": {"Growth": 0.45, "Core": 0.40, "Defensive": 0.15},
        "Aggressive": {"Growth": 0.75, "Core": 0.20, "Defensive": 0.05}
    }
    blueprint = ideal_mapping.get(risk_level, ideal_mapping["Balanced"])
    
    current_structure = {"Growth": curr_g_w, "Defensive": sum(assets[tk]["weight"] for tk in defensive_tks), "Core": 0.0}
    current_structure["Core"] = 1.0 - current_structure["Growth"] - current_structure["Defensive"]

    # Generate Blueprint Actions with Ticker References
    blueprint_actions = []
    segments = {"Growth": growth_tks, "Core": core_tks, "Defensive": defensive_tks}
    for cat, target in blueprint.items():
        diff = target - current_structure[cat]
        if abs(diff) > 0.05:
            action = "Increase" if diff > 0 else "Trim"
            members = ", ".join(segments[cat]) if segments[cat] else "None held"
            blueprint_actions.append({
                "category": cat, 
                "action": action, 
                "impact": f"{abs(diff)*100:.1f}% shift required",
                "tickers": segments[cat]
            })

    # Hedge optimizer — check existing gold/defensive before recommending more GLD
    existing_gold_w = sum(a["weight"] for tk, a in assets.items() if tk.replace('_', '.').split('.')[0].upper() in _GOLD_BASES)
    if existing_gold_w >= 0.10:
        hedge_opt = {"optimal_gld_weight": 0.0, "hedge_efficiency": "SUFFICIENT",
                     "note": f"Gold at {existing_gold_w*100:.0f}% — adequate. Consider IGLT.L for duration hedge."}
    else:
        needed = max(0.0, round(0.12 - existing_gold_w, 2))
        hedge_opt = {"optimal_gld_weight": needed, "hedge_efficiency": "HIGH" if beta > 1.1 else "LOW", "note": None}

    return {
        "scenarios": {"nasdaq_10": round(total_val*beta*-0.1, 2), "fx_5pct_shock": round(non_base_val*0.05, 2)},
        "rebalancing": {"current": round(curr_g_w, 4), "drift": round(drift, 4), "trade": trade_suggestion, "projected_beta": round(beta - (drift*0.2), 2), "projected_sharpe": None, "transaction_cost_estimate": cost},
        "attribution": {"factor": "Growth Dominant" if curr_g_w > 0.5 else "Core/Value", "div_ratio": div_ratio, "fx_exposure_risk": "HIGH" if non_base_val/total_val > 0.7 else "LOW"},
        "div_safety": div_safety,
        "kill_switch": kill_switch,
        "hedge_optimizer": hedge_opt,
        "real_cagr": real_cagr,
        "outlook_score": sum((2 if a.get("sentiment", {}).get("label") == "positive" else -2 if a.get("sentiment", {}).get("label") == "negative" else 0) for a in assets.values()),
        "ideal_blueprint": {"target": blueprint, "current": current_structure, "actions": blueprint_actions, "segments": segments}
    }

def _get_market_outlook(assets, portfolio):
    points = [
        {
            "l": "Macro Regime", "v": "Fiscal Dominance & Rate Floor", 
            "d": "We have entered a regime of 'Fiscal Dominance' where government deficit spending outpaces monetary tightening. This creates a 'Higher for Longer' floor on interest rates, significantly raising the cost of capital. In this environment, the 'Equity Risk Premium' (ERP) has compressed, making fundamental margin durability and internal cash-flow generation the only reliable factors for long-term outperformance. Success now requires identified companies that can clear a 5%+ risk-free hurdle while maintaining a positive Information Ratio.", 
            "i": "📈"
        },
        {
            "l": "Sector Rotation", "v": "Physical AI & Power Grid", 
            "d": "The AI trade is rapidly evolving from the 'Training Phase' (GPUs/Nvidia) to the 'Inference & Physicals Phase.' Capital is aggressively migrating into 'Utility Enablers'—companies specialized in data center power density, liquid cooling (e.g., Vertiv), and electrical grid modernization (e.g., Eaton). We expect a surge in demand for base-load energy sources, including nuclear SMR infrastructure, to sustain the exponential AI compute requirements of the next decade.", 
            "i": "🚀"
        },
        {
            "l": "Regional Alpha", "v": "ASEAN Corridor Alpha", 
            "d": "Global deglobalization is accelerating. As the 'China + 1' strategy matures, permanent alpha channels are opening in 'Friend-Shoring' hubs such as the ASEAN corridors (Vietnam/Thailand) and Mexico. Regional leaders in these zones are capturing a 'Geopolitical Risk Discount,' gaining market share from legacy globalists who are trapped in the crossfire of trade fragmentation and rising tariff barriers.", 
            "i": "🌍"
        },
        {
            "l": "Valuation Pivot", "v": "WACC Sensitivity & FCF Yield", 
            "d": "The era of 'Growth at Any Price' is dead. The market has pivoted to 'FCF Yield over Multiple Expansion.' With a higher Weighted Average Cost of Capital (WACC), the market is aggressively discounting 'Terminal Value' in DCF models. Institutional capital is now penalizing cash-burning names and prioritizing 'Quality Growth'—companies that can self-fund expansion while maintaining 20%+ FCF margins without relying on expensive capital markets.", 
            "i": "💎"
        },
        {
            "l": "Strategic Hedge", "v": "Hard Assets & Credit Volatility", 
            "d": "Hard Assets are no longer optional. Gold and Physical Commodities provide a structural 'Safe Haven' against currency debasement and systemic credit events. In a world of weaponized dollar-hegemony, these assets maintain low-to-negative correlations to tech-heavy portfolios, providing a vital 'Volatility Floor' and asymmetric payoff potential during non-linear tail-risk liquidation events.", 
            "i": "🛡️"
        }
    ]
    comm = []
    sharpe = portfolio.get("sharpe_ratio", 0)
    if sharpe > 1.2: 
        comm.append("Exceptional risk-adjusted harvesting detected. Your allocation is efficiently capturing alpha while successfully suppressing 'Factor Crowding' noise. This asset mix is mathematically optimized to extract returns from market swings without incurring the non-linear drawdown risk typically associated with unhedged high-beta portfolios.")
    elif sharpe < 0.7: 
        comm.append("Suboptimal efficiency detected. Your portfolio is currently exhibiting 'Noisy Volatility'—taking on significant price swings that are not being compensated by equivalent realized returns. This usually indicates 'Overlapping Factor Correlation' or 'Diworsification,' where multiple assets respond identically to interest rate or liquidity shocks, effectively destroying the benefits of traditional diversification.")
    
    future_outlook = "We anticipate a structural 'Broadening Trade.' As mega-cap tech valuations reach historical exhaustion points and face 'Multiple Compression,' institutional capital will rotate into 'Undervalued Quality.' This migration will likely favor secondary SaaS players with proven unit economics and industrial energy infrastructure providers poised to profit from the massive Western re-industrialization and AI-driven power reflation cycle."
    
    return {"points": points, "commentary": " ".join(comm), "future_outlook": future_outlook, "_source": "static_fallback"}

def _get_global_events():
    """Generate approximate economic calendar based on known recurring schedules."""
    now = datetime.datetime.now()
    year, month = now.year, now.month
    events = []

    def _nth_weekday(y, m, weekday, n):
        """Return date of the nth weekday (0=Mon) of month m in year y."""
        first_day = calendar.weekday(y, m, 1)
        offset = (weekday - first_day) % 7
        day = 1 + offset + 7 * (n - 1)
        return datetime.datetime(y, m, day)

    # FOMC: 3rd Wednesday of Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    for m in fomc_months:
        y = year if m >= month else year + 1
        try:
            dt = _nth_weekday(y, m, 2, 3)  # 3rd Wednesday
            if dt > now:
                events.append({"d": dt.strftime("%Y-%m-%d"), "e": "FOMC Rate Decision",
                               "i": "Fed Funds rate decision. Key driver for all risk assets."})
                break
        except ValueError:
            continue

    # BoE MPC: 1st Thursday of Feb, Mar, May, Jun, Aug, Sep, Nov, Dec
    boe_months = [2, 3, 5, 6, 8, 9, 11, 12]
    for m in boe_months:
        y = year if m >= month else year + 1
        try:
            dt = _nth_weekday(y, m, 3, 1)  # 1st Thursday
            if dt > now:
                events.append({"d": dt.strftime("%Y-%m-%d"), "e": "BoE MPC Meeting",
                               "i": "Bank Rate decision. GBP and gilt driver."})
                break
        except ValueError:
            continue

    # US CPI: typically 12th-14th of each month
    cpi_m = month + 1 if now.day > 15 else month
    cpi_y = year
    if cpi_m > 12:
        cpi_m, cpi_y = 1, year + 1
    events.append({"d": f"{cpi_y}-{cpi_m:02d}-13", "e": "US CPI Data",
                   "i": "Core CPI drives Fed rate expectations. Inflation trigger."})

    # US NFP: 1st Friday of each month
    nfp_m = month + 1 if now.day > 7 else month
    nfp_y = year
    if nfp_m > 12:
        nfp_m, nfp_y = 1, year + 1
    try:
        nfp_dt = _nth_weekday(nfp_y, nfp_m, 4, 1)  # 1st Friday
        events.append({"d": nfp_dt.strftime("%Y-%m-%d"), "e": "US Non-Farm Payrolls",
                       "i": "Labour market health. Affects rate cut timeline."})
    except ValueError:
        pass

    # UK Budget/Spring Statement: typically late March or late October
    budget_m = 3 if month <= 3 else 10 if month <= 10 else 3
    budget_y = year if budget_m >= month else year + 1
    events.append({"d": f"{budget_y}-{budget_m:02d}-26", "e": "UK Fiscal Statement",
                   "i": "Chancellor's fiscal policy. Gilt and GBP catalyst."})

    events.sort(key=lambda x: x["d"])
    return [e for e in events if e["d"] >= now.strftime("%Y-%m-%d")][:5]

def _generate_recommendations(assets, portfolio, risk_level, base_currency="GBP"):
    recs = []
    is_ucits = base_currency in ("GBP", "EUR") or any(tk.replace('_', '.').endswith('.L') or tk.replace('_', '.').endswith('.AS') or tk.replace('_', '.').endswith('.DE') for tk in assets)
    pb = portfolio.get("portfolio_beta", 1.0)
    sharpe = portfolio.get("sharpe_ratio", 0.0)
    yld = portfolio.get("weighted_dividend_yield", 0.0)
    
    # Beta Optimization
    if pb > 1.2:
        recs.append({
            "type": "warning", "icon": "📉", "title": "High Beta Skew", 
            "detail": f"Beta {pb:.2f} is Aggressive.", 
            "rationale": "Sensitivity to S&P 500 is excessive for current mandate. Higher risk of non-linear drawdown.", 
            "action": "Rotate to VIG/GLD"
        })
    elif pb < 0.7:
        recs.append({
            "type": "suggestion", "icon": "📈", "title": "Beta Lag", 
            "detail": f"Beta {pb:.2f} is Defensive.", 
            "rationale": "Portfolio may significantly lag in bull markets. Room for controlled growth expansion.", 
            "action": "Increase QQQ/SPY"
        })

    # Sharpe / Efficiency Optimization
    if sharpe < 0.8:
        recs.append({
            "type": "warning", "icon": "⚖️", "title": "Efficiency Gap", 
            "detail": f"Sharpe {sharpe:.2f} is Suboptimal.", 
            "rationale": "Taking too much volatility for the realized return. Portfolio is 'noisy'.", 
            "action": "Add Quality Factor"
        })
    elif sharpe > 1.5:
        recs.append({
            "type": "success", "icon": "💎", "title": "Apex Efficiency", 
            "detail": f"Sharpe {sharpe:.2f} is Exceptional.", 
            "rationale": "Harvesting maximum alpha per unit of risk. Maintain current factor weights.", 
            "action": "HODL"
        })

    # Income & Diversification
    if yld < 0.02:
        income_etf = "WQDV.L" if is_ucits else "SCHD"
        recs.append({"type": "suggestion", "icon": "💸", "title": "Income Floor", "detail": "Yield < 2%.", "rationale": "Dividends act as a structural shock absorber during flat market regimes. UCITS-compliant dividend ETF preferred for UK/EU domicile.", "action": f"Add {income_etf}"})
    
    if len(assets) < 4:
        recs.append({"type": "warning", "icon": "🏗️", "title": "Factor Concentration", "detail": "Low ticker breadth.", "rationale": "Exposure to idiosyncratic company-specific risk is too high.", "action": "Add 2+ Assets"})

    # UK GIA / Acc Fund ERI Tax Advisory
    if base_currency == "GBP":
        acc_gia_tks = [tk for tk, a in assets.items() 
                       if "acc" in (a.get("name") or "").lower() 
                       and a.get("account_type") == "GIA"]
        if acc_gia_tks:
            recs.append({
                "type": "warning", "icon": "🏦",
                "title": "GIA Tax Alert: ERI on Acc Funds",
                "detail": "Acc funds in GIA still taxable via ERI.",
                "rationale": (
                    "UK HMRC taxes Excess Reportable Income (ERI) from Accumulating (Acc) funds "
                    "in a GIA as dividend income — even though no cash is distributed. "
                    "Switching to an Acc share class does NOT reduce GIA dividend or income tax. "
                    "Recommended action: Bed & ISA transfer at UK tax year start (before 5 April) "
                    f"to shelter gains inside an ISA. Affected: {', '.join(acc_gia_tks[:3])}."
                ),
                "action": "Bed & ISA Transfer",
            })

    return recs[:6]

def _extract_events(info: dict) -> Dict[str, Any]:
    """Return only future-dated events; skip epoch (0) and stale past dates."""
    now_ts = time.time()

    def _ts(val) -> Optional[str]:
        try:
            ts = int(val)
            # Reject epoch, zero, or events more than 1 day in the past
            if ts <= 86_400 or ts < (now_ts - 86_400):
                return None
            return datetime.datetime.fromtimestamp(
                ts, tz=datetime.timezone.utc
            ).strftime("%b %d, %Y")
        except Exception:
            return None

    result: Dict[str, Any] = {}
    # Earnings — multiple yfinance fields in priority order
    earnings = _ts(
        info.get("earningsTimestamp")
        or info.get("earningsTimestampStart")
        or info.get("earningsTimestampEnd")
    )
    if earnings:
        result["earnings_date"] = earnings
    ex_div = _ts(info.get("exDividendDate"))
    if ex_div:
        result["ex_dividend_date"] = ex_div
    div_pay = _ts(info.get("dividendDate"))
    if div_pay:
        result["dividend_date"] = div_pay
    return result

# ── Tactical Asset Allocation Engine ─────────────────────────────────────────

def _mock_rsi(ticker: str) -> Optional[float]:
    """Fallback when OHLC data unavailable. Returns None (not a fake number)."""
    return None


def _mock_sentiment(ticker: str) -> Dict[str, Any]:
    """Fallback when FinBERT API is unavailable. Returns neutral with 'unavailable' flag."""
    return {
        "bullish": 0.33, "bearish": 0.33, "label": "unavailable",
        "headline_count": 0,
        "headlines": ["Sentiment data unavailable — FinBERT API offline or no recent headlines found."],
        "sentiment_delta": 0.0,
        "exhaustion_alert": False,
        "_mock": True,
    }


_TECH_S    = {"EQQQ","CNDX","QQQ","QQQM","SMH","SOXX","AINF","DAGB","MAGS","ARKK","WTEC","LGQG","IIND"}
_GOLD_TAA  = {"SGLN","GLD","IAU","GLDM","PHAU","HMSO","IGLN","SGLP","XGLD"}
_BOND_TAA  = {"TN28","IGLT","BND","AGG","TLT","VGLT","XGSD"}
_ENERGY_TAA = {"XOM","CVX","SHEL","BP","RDSB","OXY","NRGG","XENE","IOOG"}

def _generate_tactical_desk(holdings: Dict[str, float], assets: Dict[str, Any], portfolio: Dict[str, Any], ohlc_data: Dict = None, live_vix: float = None) -> List[Dict]:
    """
    Rule-Based Tactical Engine with Regime-Adjusted ATR Exit Strategy.
    Stop_Loss = Price - (ATR × (1 + VIX/100))  — wider in panic, tighter in calm.
    Take_Profit = Price + (3 × ATR)
    """
    total_val = portfolio.get("total_value", 0)
    ticker_bases = {tk.split('.')[0].upper() for tk in holdings}

    has_gold   = bool(ticker_bases & _GOLD_TAA)
    has_bonds  = bool(ticker_bases & _BOND_TAA)
    has_tech   = bool(ticker_bases & _TECH_S)
    has_energy = bool(ticker_bases & _ENERGY_TAA)

    # Prefer live quote VIX passed from main analysis
    if live_vix is not None and live_vix > 0:
        _vix = live_vix
    elif ohlc_data and "^VIX" in ohlc_data and not ohlc_data["^VIX"].empty:
        _vix = float(ohlc_data["^VIX"]["Close"].iloc[-1])
    else:
        _vix = 15.0
    vix_adj = 1 + (_vix / 100)

    desk: List[Dict] = []

    def _get_real_price(tk):
        """Get live price from OHLC data or assets, not hardcoded."""
        api_tk = tk.split('_')[0]
        if ohlc_data and api_tk in ohlc_data and not ohlc_data[api_tk].empty:
            return float(ohlc_data[api_tk]['Close'].iloc[-1])
        return assets.get(tk, {}).get("price", 0)

    def _get_exit_strategy(tk, current_price):
        hist_df = ohlc_data.get(tk.split('_')[0], pd.DataFrame()) if ohlc_data else pd.DataFrame()
        atr = _compute_atr(hist_df) if not hist_df.empty else (0.02 * current_price)
        # Regime-adjusted: wider stops when VIX elevated
        sl = round(current_price - (atr * vix_adj), 2)
        tp = round(current_price + (3 * atr), 2)
        risk = round(current_price - sl, 2)
        reward = round(tp - current_price, 2)
        rr = round(reward / risk, 1) if risk > 0 else 0
        qty_suggestion = max(1, round(total_val * 0.02 / current_price)) if current_price > 0 else 0
        return {
            "stop_loss": sl,
            "take_profit": tp,
            "entry_price": round(current_price, 2),
            "risk_reward": f"1:{rr}",
            "vix_adjustment": round(vix_adj, 2),
            "atr_pct": round(atr / current_price * 100, 2) if current_price > 0 else 0,
            "suggested_qty": qty_suggestion,
            "rationale": f"Regime-Adjusted ATR exit (VIX {_vix:.0f}, multiplier {vix_adj:.2f}×). Stop = Price - ATR×{vix_adj:.2f}, TP = Price + 3×ATR. R/R = 1:{rr}."
        }

    # Signal 1 — GEO-RISK: portfolio missing energy exposure → suggest hedge
    if not has_energy:
        price = _get_real_price("NRGG.L")
        if price > 0:
            desk.append({
                "severity": "GEO-RISK", "icon": "🛢️",
                "event": "Portfolio Missing Energy Exposure — Sector Gap Detected",
                "action": "Buy", "ticker": "NRGG.L",
                "amount": round(total_val * 0.03),
                "exit_strategy": _get_exit_strategy("NRGG.L", price),
                "rationale": (
                    "Your portfolio has zero energy sector allocation. "
                    "iShares Oil & Gas Exploration & Production UCITS ETF (NRGG.L) provides direct "
                    "exposure to upstream E&P companies and hedges against energy price shocks."
                ),
                "time_horizon": "72h",
                "_source": "rule_engine",
            })

    # Signal 2 — MACRO-SHOCK: high tech concentration → suggest trimming
    tech_tickers = [tk for tk in holdings if tk.split('.')[0].upper() in _TECH_S]
    if has_tech and tech_tickers:
        top_tech = max(tech_tickers, key=lambda tk: assets.get(tk, {}).get("value", 0))
        price = assets.get(top_tech, {}).get("price", 0) or _get_real_price(top_tech)
        trim_val = round(assets.get(top_tech, {}).get("value", 0) * 0.10)
        if trim_val > 0 and price > 0:
            desk.append({
                "severity": "MACRO-SHOCK", "icon": "🌡️",
                "event": "High-Weight Tech Concentration — Duration Risk Elevated",
                "action": "Trim", "ticker": top_tech,
                "amount": trim_val,
                "exit_strategy": _get_exit_strategy(top_tech, price),
                "rationale": (
                    f"Long-duration growth equities ({top_tech}) are your largest position. "
                    "Trimming 10% reduces concentration risk and discount-rate sensitivity."
                ),
                "time_horizon": "72h",
                "_source": "rule_engine",
            })

    # Signal 3 — MOMENTUM: portfolio missing gold hedge → suggest adding
    if not has_gold:
        price = _get_real_price("SGLN.L")
        if price > 0:
            desk.append({
                "severity": "MOMENTUM", "icon": "🏅",
                "event": "Portfolio Missing Gold Hedge — Tail-Risk Unhedged",
                "action": "Buy", "ticker": "SGLN.L",
                "amount": round(total_val * 0.02),
                "exit_strategy": _get_exit_strategy("SGLN.L", price),
                "rationale": (
                    "Your portfolio has no gold allocation. "
                    "SGLN.L adds portfolio convexity against macro tail risks and inflation."
                ),
                "time_horizon": "72h",
                "_source": "rule_engine",
            })

    return desk[:3]


def _compute_tactical_blueprint(ideal_blueprint: Dict, tactical_desk: List[Dict]) -> Dict:
    """
    Overlays tactical tilts on top of strategic blueprint targets
    based on active catalyst signals from the tactical desk.
    """
    base = dict(ideal_blueprint.get("target", {"Growth": 0.5, "Core": 0.35, "Defensive": 0.15}))
    tilt = {"Growth": 0.0, "Core": 0.0, "Defensive": 0.0}
    rationale_parts: List[str] = []

    for alert in tactical_desk:
        sev    = alert.get("severity", "")
        action = alert.get("action", "")
        if sev == "GEO-RISK" and tilt["Defensive"] == 0:
            tilt["Defensive"] += 0.05
            tilt["Growth"]    -= 0.05
            rationale_parts.append("+5% Defensive tilt — geopolitical risk surge")
        elif sev == "MACRO-SHOCK" and action == "Trim" and tilt["Core"] == 0:
            tilt["Growth"] -= 0.05
            tilt["Core"]   += 0.05
            rationale_parts.append("+5% Core / −5% Growth — macro shock re-pricing")
        elif sev == "MOMENTUM" and tilt["Defensive"] < 0.04:
            tilt["Defensive"] += 0.02
            tilt["Core"]      -= 0.02
            rationale_parts.append("+2% Defensive overlay — gold momentum signal")

    tactical = {cat: round(max(0.0, min(1.0, base.get(cat, 0) + tilt[cat])), 3) for cat in base}
    total = sum(tactical.values())
    if total > 0:
        tactical = {k: round(v / total, 3) for k, v in tactical.items()}

    return {
        "target": tactical,
        "rationale": " · ".join(dict.fromkeys(rationale_parts)) or "No tactical tilt active — strategic targets unchanged.",
        "active": len(rationale_parts) > 0,
    }


# ── Entity map: raw ETF ticker base → macro/thematic search keywords ──────────
# Querying the News API with "EQQQ.L" returns Italian bank articles;
# "Nasdaq 100" returns actionable US tech macro news.
_ENTITY_MAP: Dict[str, List[str]] = {
    # Broad-market & global
    "VWRP": ["global equities", "MSCI World", "global stock market"],
    "SWRD": ["MSCI World", "developed markets equities"],
    "VUSA": ["S&P 500", "US equities", "Federal Reserve"],
    "CSPX": ["S&P 500", "US equities"],
    "SPY":  ["S&P 500", "US stock market", "Federal Reserve"],
    # Tech / Nasdaq
    "EQQQ": ["Nasdaq 100", "US tech sector", "AI stocks"],
    "QQQ":  ["Nasdaq 100", "US tech sector", "AI stocks"],
    "IITU": ["technology sector", "semiconductor stocks", "AI equities"],
    # Emerging / regional
    "EMIM": ["emerging markets", "China economy", "EM equities"],
    "VFEM": ["emerging markets", "developing economies"],
    "IIND": ["India economy", "Nifty 50", "India equities"],
    "ISJP": ["Japan equities", "Nikkei", "Bank of Japan"],
    "WLDS": ["global small cap", "Russell 2000", "small cap equities"],
    # UK & Europe
    "ISF":  ["FTSE 100", "UK equities", "UK economy"],
    "VMID": ["FTSE 250", "UK mid-cap", "UK economy"],
    "VERX": ["European equities", "Eurozone economy", "ECB"],
    # Fixed income
    "VAGP": ["global bond market", "interest rates", "bond yields"],
    "VGOV": ["UK gilts", "Bank of England", "UK government bonds"],
    "IGLH": ["corporate bonds", "credit spreads", "investment grade"],
    "DAGB": ["global bond market", "aggregate bonds"],
    "VJPB": ["Japan bonds", "Bank of Japan", "yen"],
    "TLT":  ["US Treasury bonds", "long duration bonds", "Federal Reserve"],
    "BND":  ["US bond market", "Treasury yields"],
    "AGG":  ["US bond market", "bond yields"],
    # Commodities / alternatives
    "SGLN": ["gold prices", "gold commodities", "gold bullion"],
    "PHGP": ["gold prices", "precious metals"],
    "GLD":  ["gold prices", "gold commodities"],
    "NRGG": ["oil gas sector", "energy stocks", "oil prices"],
    "AINF": ["infrastructure assets", "real assets", "inflation hedge"],
    "CMOP": ["commodity markets", "raw materials"],
    "SPOG": ["real estate", "property sector", "REITs"],
    # Defence / thematic
    "DFNG": ["defense sector", "geopolitics", "military spending"],
    "DFEN": ["aerospace defense", "US defense spending"],
    # Income
    "SDIP": ["dividend stocks", "income equities"],
    # Fund specific entities
    "0P0000XW0J": ["Invesco UK", "UK high income", "Invesco High Income"],
    "0P0000X63C": ["Jupiter India", "India economy", "Jupiter Asset Management"],
}

# ── Junk-headline patterns to strip retail SEO chaff ─────────────────────────
_NEWS_JUNK_PATTERNS = [
    "most popular", "most bought", "top stocks for",
    "stocks to watch", "best etfs to buy", "etfs to buy",
    "should you buy", "worth buying",
]

# ── 7-day cutoff (epoch seconds) ─────────────────────────────────────────────
def _news_cutoff_ts() -> float:
    """Return timestamp for 24 hours ago."""
    return time.time() - 86_400


async def _get_sentiment(ticker: str) -> Dict[str, Any]:
    if ticker in _sentiment_cache:
        c = _sentiment_cache[ticker]
        if time.time() - c["ts"] < CACHE_TTL: return c["data"]

    # Strip composite suffix (e.g. AAPL_ISA -> AAPL) for API calls
    api_ticker = ticker.split('_')[0]
    clean_tk = api_ticker.split(".")[0].upper()
    search_terms = _ENTITY_MAP.get(clean_tk, [api_ticker])   # fallback: raw ticker
    cutoff = _news_cutoff_ts()

    async def fetch_headlines(query: str) -> List[str]:
        try:
            news = await _in_thread(lambda: yf.Ticker(query).news)
            if not news:
                return []
            results: List[str] = []
            for n in news[:15]:
                content = n.get("content", {})
                title = content.get("title", "")
                if not title:
                    continue
                # ── 7-day recency filter ──────────────────────────────────
                pub = content.get("pubDate") or content.get("displayTime") or ""
                if pub:
                    try:
                        ts = datetime.datetime.fromisoformat(
                            pub.rstrip("Z")
                        ).replace(tzinfo=datetime.timezone.utc).timestamp()
                        if ts < cutoff:
                            continue
                    except Exception:
                        pass  # unparseable date — keep the headline
                # ── Retail-SEO junk filter ────────────────────────────────
                tl = title.lower()
                if any(pat in tl for pat in _NEWS_JUNK_PATTERNS):
                    continue
                results.append(title)
            return results
        except Exception:
            return []

    # Fetch from each mapped entity term (up to 3), then deduplicate
    headlines: List[str] = []
    for term in search_terms[:3]:
        headlines.extend(await fetch_headlines(term))

    # Fallback: try the raw ticker if entity search yielded nothing
    if not headlines:
        headlines = await fetch_headlines(api_ticker)

    if not headlines: return _mock_sentiment(ticker)
    
    headlines = list(dict.fromkeys(headlines))[:20]
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(FINBERT_API_URL, headers=headers, json={"inputs": headlines}, timeout=30.0)
            results = resp.json(); total_b = total_r = 0.0
            if isinstance(results, dict) and "error" in results: raise Exception(results["error"])
            for res in results:
                scores = res if isinstance(res, list) else [res]
                for s in scores:
                    if s.get("label") == "positive": total_b += s.get("score", 0)
                    elif s.get("label") == "negative": total_r += s.get("score", 0)
            avg_b, avg_r = round(total_b/max(len(headlines),1), 3), round(total_r/max(len(headlines),1), 3)
            data = {"bullish": avg_b, "bearish": avg_r, "label": "positive" if avg_b > avg_r and avg_b > 0.35 else "negative" if avg_r > avg_b and avg_r > 0.35 else "neutral", "headline_count": len(headlines), "headlines": headlines, "sentiment_delta": round(avg_b - 0.45, 2), "exhaustion_alert": avg_b < 0.25}
            _sentiment_cache[ticker] = {"ts": time.time(), "data": data}
            return data
    except:
        s = _mock_sentiment(ticker)
        s["headline_count"] = len(headlines)
        s["headlines"]      = headlines
        return s


# ── Live headline fetcher (replaces fake _HEADLINE_POOL) ─────────────────────
async def _fetch_live_headlines(n: int = 10) -> List[tuple]:
    """Fetch real market headlines from yfinance for broad macro and futures."""
    macro_tickers = ["SPY", "QQQ", "GLD", "TLT", "USO", "GC=F", "ES=F", "CL=F", "BTC-USD", "^FTSE"]
    tag_map = {
        "SPY": "EQUITY-MACRO", "QQQ": "TECH-MOMENTUM", "GLD": "SAFE-HAVEN",
        "TLT": "RATES-SHOCK", "USO": "ENERGY-GEO", "GC=F": "SAFE-HAVEN",
        "ES=F": "FUTURES-PULSE", "CL=F": "ENERGY-GEO", "BTC-USD": "RISK-ON",
        "^FTSE": "UK-EXPOSURE"
    }
    headlines: List[tuple] = []
    cutoff = _news_cutoff_ts()
    for tk in macro_tickers:
        try:
            news = await _in_thread(lambda t=tk: yf.Ticker(t).news)
            for item in (news or [])[:5]:
                content = item.get("content", {})
                title = content.get("title", "")
                if not title:
                    continue
                pub = content.get("pubDate") or content.get("displayTime") or ""
                if pub:
                    try:
                        ts = datetime.datetime.fromisoformat(
                            pub.rstrip("Z")
                        ).replace(tzinfo=datetime.timezone.utc).timestamp()
                        if ts < cutoff:
                            continue
                    except Exception:
                        pass
                tl = title.lower()
                if any(p in tl for p in _NEWS_JUNK_PATTERNS):
                    continue
                headlines.append((title, tag_map.get(tk, "SECTOR-ROTATE")))
        except Exception:
            continue
    # Deduplicate by headline text
    seen = set()
    unique: List[tuple] = []
    for hl, tag in headlines:
        if hl not in seen:
            seen.add(hl)
            unique.append((hl, tag))
    if len(unique) < 3:
        return []  # Not enough real headlines → LLM will rely on portfolio data only
    return unique[:n]

async def _fetch_futures_pulse() -> str:
    """Fetch price and % change for key futures to give LLM 'Overnight Pulse' context."""
    symbols = {
        "ES=F": "S&P 500 Futures",
        "NQ=F": "Nasdaq 100 Futures",
        "CL=F": "Crude Oil Futures",
        "GC=F": "Gold Futures",
        "^FTSE": "FTSE 100 Index",
        "BTC-USD": "Bitcoin"
    }
    pulse_lines = []
    try:
        data = await _in_thread(lambda: yf.download(list(symbols.keys()), period="2d", interval="1d", progress=False, group_by='ticker'))
        for sym, name in symbols.items():
            try:
                s_data = data[sym] if sym in data else None
                if s_data is not None and len(s_data) >= 2:
                    last_close = float(s_data['Close'].iloc[-2])
                    curr_price = float(s_data['Close'].iloc[-1])
                    chg = (curr_price / last_close - 1) * 100
                    # Standardize currency formatting for prompt
                    fmt_price = f"${curr_price:,.2f}" if sym != "^FTSE" else f"{curr_price:,.0f} pts"
                    pulse_lines.append(f"  - {name}: {fmt_price} ({chg:+.2f}%)")
            except: continue
    except: pass
    return "\n".join(pulse_lines) if pulse_lines else "  [Live pulse data currently unavailable]"

async def _call_cio_llm(
    holdings: Dict[str, float],
    assets: Dict[str, Any],
    portfolio: Dict[str, Any],
    ideal_blueprint: Dict[str, float],
    alpha_alerts: List[Dict] = None,
    apex_advisor: Dict[str, Any] = None,
    advanced_intel: Dict[str, Any] = None,
    rsi_cache: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Calls Claude (CIO persona) to generate live TAA recommendations by
    synthesizing Alpha Alerts, Apex Advisor logic, and Quant risk metrics.
    """
    # ── Build daily news context from REAL yfinance headlines ──────────────
    today = datetime.datetime.now(datetime.timezone.utc)
    # Concurrent fetch for headlines and price pulse
    todays_headlines, market_pulse = await asyncio.gather(
        _fetch_live_headlines(10),
        _fetch_futures_pulse()
    )

    # ── Format portfolio context ──────────────────────────────────────────────
    total_val  = portfolio.get("total_value", 0)
    beta       = portfolio.get("portfolio_beta", 1.0)
    vol        = portfolio.get("volatility", 0.15)
    sharpe     = portfolio.get("sharpe_ratio", 0.0)
    var_95     = portfolio.get("var_95_daily", 0.02)

    # ── Extract technical/alpha signals for the prompt ────────────────────────
    alpha_text = ""
    if alpha_alerts:
        alpha_text = "## Active Alpha Alerts (Volume/Lead-Lag)\n" + "\n".join(
            f"- {a['symbol']}: {a['title']} (Conviction {a['conviction']}%) | {a['rationale']}"
            for a in alpha_alerts[:5]
        ) + "\n\n"

    advisor_text = ""
    if apex_advisor and "scores" in apex_advisor:
        advisor_text = "## Senior Quant Advisor Signals (Matrix/Exhaustion)\n"
        for tk, log in list(apex_advisor["scores"].items())[:8]:
            advisor_text += (f"- {tk}: Strategy: {log.get('strat', 'Neutral')} | Action: {log.get('action_instruction', 'HOLD')} | "
                             f"Exhaustion: {log.get('exhaustion_status', 'Neutral')} | Reason: {log.get('why', 'No data')}\n")
        advisor_text += "\n"

    quant_text = ""
    if advanced_intel:
        quant_text = "## Institutional Quant Metrics\n"
        quant_text += f"- Diversification Ratio: {advanced_intel.get('attribution', {}).get('div_ratio', 1.0)}\n"
        quant_text += f"- Hedge Efficiency: {advanced_intel.get('hedge_optimizer', {}).get('hedge_efficiency', 'N/A')}\n\n"

    asset_lines = []
    for tk, a in assets.items():
        s   = a.get("sentiment", {})
        rsi = s.get("rsi") or (rsi_cache or {}).get(tk) or 50.0
        sig = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
        asset_lines.append(
            f"  {tk}: £{a['value']:,.0f} | {a['weight']*100:.1f}% weight | "
            f"RSI {rsi:.0f} ({sig}) | Bull {s.get('bullish', 0.33):.0%} Bear {s.get('bearish', 0.33):.0%}"
        )

    if todays_headlines:
        headline_text = "\n".join(f"  [{tag}] {hl}" for hl, tag in todays_headlines)
    else:
        headline_text = "  [No live market headlines available]"
    asset_text    = "\n".join(asset_lines)

    user_msg = (
        f"## Today's Date\n{today.strftime('%A, %d %B %Y')}\n\n"
        f"## Live Market Pulse (Overnight/Pre-market)\n{market_pulse}\n\n"
        f"## Client Portfolio\n{asset_text}\n\n"
        f"Totals: £{total_val:,.0f} (Invested) | Uninvested Cash: £{portfolio.get('total_cash', 0):,.0f} | "
        f"Beta {beta:.2f} | Vol {vol*100:.1f}% | Sharpe {sharpe:.2f} | Daily VaR95 {var_95*100:.2f}%\n\n"
        f"{alpha_text}{advisor_text}{quant_text}"
        f"## Live Market Headlines\n{headline_text}\n\n"
        f"Generate the 'Executive Master Verdict' by synthesizing all the signals above. "
        f"IMPORTANT: The 'Live Market Pulse' is your GROUND TRUTH. If headlines suggest a surge but the Pulse shows a decline, the headline is LAGGING—prioritize the Pulse data. Do not hallucinate prices or trends that contradict the Pulse section. "
        f"Sort the 10 directives by descending priority. Return ONLY JSON."
    )

    # ── Call Claude 3 Haiku ──────────────────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from environment.")

    try:
        from anthropic import AsyncAnthropic as _AsyncAnthropic
        client = _AsyncAnthropic(api_key=api_key)
        
        response = await client.messages.create(
            model="claude-opus-4-6", 
            max_tokens=8192,
            system=_CIO_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        
        # Robust extraction (handles thinking blocks and multiple text blocks)
        raw = ""
        for block in response.content:
            if block.type == "text":
                raw += block.text
        raw = raw.strip()
        print(f"[CIO LLM] Raw Response Received ({len(raw)} chars)")
    except Exception as e:
        raise RuntimeError(f"Anthropic API call failed: {str(e)}")

    # Strip accidental markdown fences
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()

    # ── Robust JSON Repair (for occasional truncation) ───────────────────────
    if not raw.endswith("}"):
        print("[CIO LLM] WARNING: Response appears truncated. Attempting repair...")
        # Close any open quotes
        if raw.count('"') % 2 != 0: raw += '"'
        # Add missing braces
        open_braces = raw.count("{") - raw.count("}")
        raw += ("}" * open_braces)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as je:
        print(f"[CIO LLM] JSON Decode Error at char {je.pos}. Raw content snippet: {raw[-100:]}")
        raise RuntimeError("CIO LLM returned invalid JSON. Check server logs.")

    # Validate schema & normalise blueprint weights
    assert isinstance(result.get("tactical_desk"), list), "missing tactical_desk"
    assert isinstance(result.get("tactical_blueprint", {}).get("target"), dict), "missing blueprint.target"
    bp_tgt = result["tactical_blueprint"]["target"]
    bp_sum = sum(bp_tgt.values())
    if bp_sum > 0:
        result["tactical_blueprint"]["target"] = {k: round(v / bp_sum, 3) for k, v in bp_tgt.items()}
    result["tactical_blueprint"].setdefault("active", len(result["tactical_desk"]) > 0)
    result["_llm_powered"] = True
    return result


async def _call_gemini_llm(user_msg: str) -> Dict[str, Any]:
    """Calls Google Gemini 2.0 Flash (Free Tier) via direct REST API (No SDK needed)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from your .env file.")

    # Google Gemini REST API Endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    payload = {
        "system_instruction": {"parts": [{"text": _CIO_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_msg}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API Error {resp.status_code}: {resp.text}")
            
            data = resp.json()
            # Extract text from Google's deep JSON structure
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"[CIO Gemini REST] Raw Response Received ({len(raw)} chars)")
    except Exception as e:
        raise RuntimeError(f"Gemini REST call failed: {str(e)}")

    # Strip accidental markdown fences
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()

    # Simple repair
    if not raw.endswith("}"):
        raw += "}" * (raw.count("{") - raw.count("}"))

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[CIO Gemini] JSON Decode Error. Raw: {raw}")
        raise RuntimeError("Gemini returned invalid JSON. Try again.")

    # Normalise weights
    bp_tgt = result.get("tactical_blueprint", {}).get("target", {})
    bp_sum = sum(bp_tgt.values()) if bp_tgt else 0
    if bp_sum > 0:
        result["tactical_blueprint"]["target"] = {k: round(v / bp_sum, 3) for k, v in bp_tgt.items()}
    
    result["_llm_powered"] = True
    return result


async def _cio_with_fallback(
    holdings: Dict[str, float],
    assets: Dict[str, Any],
    portfolio: Dict[str, Any],
    ideal_blueprint: Dict[str, float],
    alpha_alerts: List[Dict] = None,
    apex_advisor: Dict[str, Any] = None,
    advanced_intel: Dict[str, Any] = None,
    rsi_cache: Dict[str, float] = None,
    live_vix: float = None,
    model_provider: str = "claude",
) -> Dict[str, Any]:
    """Calls the chosen LLM; returns error state if LLM fails."""
    try:
        if model_provider == "gemini":
            # For Gemini, we pass the user_msg which we re-generate inside or pass out
            # To keep it DRY, we'll refactor slightly or just pass the logic
            # Let's just generate the msg here to avoid complex refactors
            # (Note: In a real app we'd split the prompt builder from the caller)
            from datetime import datetime as _dt
            today = _dt.now(datetime.timezone.utc)
            # Re-generate context for Gemini
            alpha_text = ""
            if alpha_alerts:
                alpha_text = "## Active Alpha Alerts\n" + "\n".join([f"- {a['symbol']}: {a['title']}" for a in alpha_alerts[:5]]) + "\n\n"
            
            asset_text = "\n".join([f"  {tk}: £{a['value']:,.0f} | {a['weight']*100:.1f}% weight" for tk, a in assets.items()])
            
            user_msg = (
                f"## Today's Date\n{today.strftime('%A, %d %B %Y')}\n\n"
                f"## Client Portfolio\n{asset_text}\n\n"
                f"Totals: £{portfolio.get('total_value', 0):,.0f} | Uninvested Cash: £{portfolio.get('total_cash', 0):,.0f}\n\n"
                f"{alpha_text}"
                f"Generate the 'Executive Master Verdict' synthesized from signals. Return ONLY JSON."
            )
            return await _call_gemini_llm(user_msg)
        else:
            return await _call_cio_llm(
                holdings, assets, portfolio, ideal_blueprint, 
                alpha_alerts=alpha_alerts, 
                apex_advisor=apex_advisor, 
                advanced_intel=advanced_intel,
                rsi_cache=rsi_cache
            )
    except Exception as e:
        traceback.print_exc()
        err_msg = str(e)
        print(f"[CIO LLM] ERROR: {err_msg}")
        return {
            "master_verdict": {
                "summary": "The Intelligence Engine encountered a connection error.",
                "directives": [
                    {
                        "priority": "CRITICAL",
                        "action": "ERROR: Intelligence Engine Unavailable",
                        "timeframe": "N/A",
                        "rationale": f"System error during analysis: {err_msg}. Review manual risk parameters."
                    }
                ]
            },
            "tactical_desk": [],
            "tactical_blueprint": {"target": ideal_blueprint.get("target", {}), "rationale": "Strategic mandate active (LLM Error)"},
            "summary_hints": {"var": "Error", "beta": "Error", "sharpe": "Error", "cagr": "Error"},
            "quant_intelligence": "AI analysis unavailable.",
            "market_outlook": {"points": []},
            "sentiment_updates": {},
            "_llm_powered": False,
            "_llm_error": err_msg
        }


@app.get("/api/config")
async def get_config():
    return {
        "apiKey": os.getenv("FB_API_KEY"),
        "authDomain": os.getenv("FB_AUTH_DOMAIN"),
        "projectId": os.getenv("FB_PROJECT_ID"),
        "storageBucket": os.getenv("FB_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FB_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FB_APP_ID"),
        "measurementId": os.getenv("FB_MEASUREMENT_ID")
    }

async def watcher_loop():
    """Periodically checks the saved portfolio and alerts on changes."""
    interval = 1800 # 30 mins
    uid = os.getenv("VESPER_USER_ID", "default_user")
    last_verdict_actions = set()
    
    while True:
        try:
            p_data = _load_user_transactions(uid)
            # Flatten holdings for engine
            holdings = {tk: p["qty"] for tk, p in p_data.get("open_positions", {}).items()}
            if holdings:
                req = PortfolioRequest(
                    holdings=holdings,
                    risk_level="Balanced",
                    enable_cio_llm=True,
                    model_provider="gemini"
                )
                result = await run_full_analysis(req)
                mv = result.get("master_verdict", {})
                directives = mv.get("directives", [])
                
                # Check for new high-priority directives
                current_actions = {d["action"] for d in directives if d["priority"] in ["CRITICAL", "HIGH"]}
                new_criticals = current_actions - last_verdict_actions
                
                if new_criticals:
                    alert_msg = "<b>🚨 BACKGROUND ALERT: New High-Priority Directives</b>\n\n"
                    for d in directives:
                        if d["action"] in new_criticals:
                            alert_msg += f"<b>{d['priority']}</b>: {d['action']}\n<i>{d['rationale']}</i>\n\n"
                    await send_telegram_alert(alert_msg)
                    last_verdict_actions = current_actions
        except Exception as e:
            print(f"[Watcher] Loop Error: {e}")
        await asyncio.sleep(interval)

async def telegram_polling_loop():
    """Listener for /status and /verdict commands."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or ":" not in token: 
        print("[Telegram] Bot token missing or invalid format. Polling disabled.")
        return
    last_update_id = 0
    uid = os.getenv("VESPER_USER_ID", "default_user")
    
    while True:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params={"offset": last_update_id + 1, "timeout": 10})
                
                if resp.status_code == 409:
                    print("[Telegram] Conflict: Another bot instance is likely running. Retrying in 10s...")
                    await asyncio.sleep(10)
                    continue
                elif resp.status_code != 200:
                    print(f"[Telegram] Polling Error {resp.status_code}: {resp.text}")
                    await asyncio.sleep(5)
                    continue

                data = resp.json()
                if not data.get("ok"):
                    print(f"[Telegram] API Error: {data}")
                    await asyncio.sleep(5)
                    continue

                updates = data.get("result", [])
                for update in updates:
                    last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    if not text: continue

                    if text == "/status":
                        await send_telegram_alert("Vesper Engine is ACTIVE. Monitoring portfolio via Gemini.")
                    elif text == "/verdict":
                        await send_telegram_alert("🔄 Running on-demand background analysis...")
                        try:
                            p_data = _load_user_transactions(uid)
                            holdings = {tk: p["qty"] for tk, p in p_data.get("open_positions", {}).items()}
                            if holdings:
                                req = PortfolioRequest(holdings=holdings, risk_level="Balanced", enable_cio_llm=True, model_provider="gemini")
                                res = await run_full_analysis(req)
                                mv = res.get("master_verdict", {})
                                reply = f"<b>Master Verdict Summary:</b>\n{mv.get('summary', 'No summary.')}\n\n"
                                for d in mv.get("directives", [])[:3]:
                                    reply += f"• <b>{d['priority']}</b>: {d['action']}\n"
                                await send_telegram_alert(reply)
                            else:
                                await send_telegram_alert("No saved portfolio found to analyze. Please click 'Save' in the web app first.")
                        except Exception as e:
                            await send_telegram_alert(f"⚠️ Analysis failed: {str(e)}")
        except Exception as e:
            # Show the full error type if message is empty
            err_type = type(e).__name__
            print(f"[Telegram] Network/Polling Error ({err_type}): {e}")
            await asyncio.sleep(10) # Back off on network error
        await asyncio.sleep(1)

@app.post("/api/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    try:
        return await run_full_analysis(request)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

async def run_full_analysis(request: PortfolioRequest) -> Dict[str, Any]:
    """Core analysis engine — can be called by API or Background Watcher."""
    holdings = request.holdings; tickers = list(holdings.keys())
    base_curr = request.base_currency or "GBP"
    
    async def _fetch_info(ticker):
        try:
            # Strip composite suffix (e.g. AAPL_ISA -> AAPL) for API calls
            api_ticker = ticker.split('_')[0]
            api_ticker = _resolve_ticker(api_ticker)
            
            t = yf.Ticker(api_ticker)
            # yf info can be slow or throw 404s for invalid symbols
            info = await _in_thread(lambda: t.info)
            if not info or not isinstance(info, dict):
                info = {}
            p = _to_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0)
            return ticker, info, p
        except Exception as e:
            print(f"[Vesper] Info fetch failed for {ticker}: {e}")
            return ticker, {}, 0
    
    res_info = await asyncio.gather(*[_fetch_info(t) for t in tickers])
    infos = {tk: info for tk, info, p in res_info}
    
    # Currency Normalization Logic
    usd_gbp = await _get_fx_rate("USD", base_curr)
    gbp_usd = 1.0 / usd_gbp if usd_gbp else 1.0
    fallbacks = request.fallback_prices or {}
    
    assets = {}
    for tk, info, p in res_info:
        # Default to GBP for London tickers if info is missing
        default_curr = "GBP" if tk.endswith(".L") else "USD"
        curr = safe_fetch(info, "currency", default_curr)
        if curr in ("GBp", "GBX"): p /= 100.0; curr = "GBP"
        
        # Apply fallback if primary price is 0/None
        api_ticker = tk.split('_')[0]
        if (not p or p <= 0):
            if tk in fallbacks:
                p = fallbacks[tk]
            elif api_ticker in fallbacks:
                p = fallbacks[api_ticker]
            
            # If it's a London asset and the price is high (>1000), it's likely in pence
            if p is not None and api_ticker.endswith(".L") and p > 500:
                p /= 100.0
        
        # Ensure p is at least 0.0 for value calculation
        p = p or 0.0
        
        # Convert asset price to base_currency for summary
        fx_to_base = 1.0
        if curr == "USD" and base_curr == "GBP": fx_to_base = usd_gbp
        elif curr == "GBP" and base_curr == "USD": fx_to_base = gbp_usd
        
        val_in_base = round(p * holdings[tk] * fx_to_base, 2)
        
        assets[tk] = {
            "name": safe_fetch(info, "longName", tk), 
            "price": round(p, 2), 
            "quantity": holdings[tk], 
            "value": val_in_base, 
            "currency": curr,
            "account_type": request.account_mapping.get(tk, "ISA"),
            "dividend_yield": _round(info.get("dividendYield")), 
            "pe_ratio": _round(info.get("trailingPE")), 
            "beta": _round(info.get("beta")), 
            "sector": safe_fetch(info, "sector", "N/A"),
            "institutional_flow_score": round(_to_float(info.get("heldPercentInstitutions", 0.5))*100, 1),
            "ter": _round(_to_float(info.get("annualReportExpenseRatio") or info.get("totalExpenseRatio"))
                          or _KNOWN_TER.get(api_ticker.split('.')[0].upper())),
        }
    
    total_val = sum(a["value"] for a in assets.values())
    if total_val <= 0: total_val = 1.0
    for a in assets.values(): a["weight"] = round(a["value"]/total_val, 4)
    
    # Clean tickers (strip _ISA etc) for external API history fetch
    # V6.0: Ensure HG=F and BTC-USD are included for lead-lag
    api_tickers = list(set([t.split('_')[0] for t in tickers] + ["SPY", "^VIX", "VWRL.L", "HG=F", "BTC-USD"] + list(GLOBAL_WATCHLIST.keys())))
    full_prices, ohlc_data = await _fetch_full_history(api_tickers)

    # Pre-compute real RSI for all tickers using OHLC data
    _rsi_cache: Dict[str, float] = {}
    for _tk_raw in tickers:
        _api_tk = _tk_raw.split('_')[0]
        _ohlc = ohlc_data.get(_api_tk, pd.DataFrame())
        if not _ohlc.empty and 'Close' in _ohlc.columns and len(_ohlc) >= 15:
            _rsi_cache[_tk_raw] = _compute_rsi(_ohlc['Close'])
        else:
            _rsi_cache[_tk_raw] = None  # no OHLC data available

    risk_data = _compute_risk_history_from_prices(full_prices, tickers, {tk: assets[tk]["weight"] for tk in tickers})
    
    # Parallel sentiment and V6.0 alpha signal analysis
    sentiments = await asyncio.gather(*[_get_sentiment(t) for t in tickers])
    # Fetch latest VIX: prefer live quote (regularMarketPrice) over historical close
    _vix_available = False
    _vix_live = 15.0
    try:
        _vix_info = await _in_thread(lambda: yf.Ticker("^VIX").info)
        _vix_rt = _to_float(_vix_info.get("regularMarketPrice"))
        if _vix_rt and _vix_rt > 0:
            _vix_live = _vix_rt
            _vix_available = True
    except Exception:
        pass
    if not _vix_available and "^VIX" in full_prices.columns and not full_prices["^VIX"].dropna().empty:
        _vix_live = float(full_prices["^VIX"].dropna().iloc[-1])
        _vix_available = True
    alpha_alerts = await get_lead_lag_signals(holdings, ohlc_data, live_vix=_vix_live)
    alpha_alerts.extend(get_institutional_flow(holdings, ohlc_data=ohlc_data, vix=_vix_live))

    for i, (tk, s) in enumerate(zip(tickers, sentiments)):
        rsi_val = _rsi_cache.get(tk)
        if rsi_val is None:
            rsi_val = 50.0  # neutral fallback — no OHLC data
        s["rsi"] = rsi_val
        s["rsi_signal"] = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL"
        assets[tk]["sentiment"] = s
        # Map V6.0 alerts back to assets for rendering
        assets[tk]["alpha_alerts"] = [a for a in alpha_alerts if a["symbol"] == tk]
        assets[tk]["risk_contribution_pct"] = round(risk_data.get("risk_contributions", {}).get(tk, 0) * 100, 2)

    blended_ter = sum((_to_float(a.get("ter")) or _KNOWN_TER.get(tk.replace('_', '.').split('.')[0].upper(), 0.002)) * a["weight"]
                      for tk, a in assets.items())
    total_cash_val = sum(request.cash_balances.values()) if request.cash_balances else 0.0
    p_summary = {**risk_data["portfolio_risk"], "total_value": round(total_val, 2), "total_cash": round(total_cash_val, 2), "weighted_dividend_yield": round(sum((_to_float(a["dividend_yield"]) or 0)*a["weight"] for a in assets.values()), 4), "portfolio_beta": round(sum((_to_float(a["beta"]) or 1)*a["weight"] for a in assets.values()), 2), "asset_count": len(tickers), "currency": base_curr, "blended_ter": round(blended_ter, 4), "vix_assumed": not _vix_available, "risk_free_rate": RISK_FREE_RATE}
    advanced = _get_advanced_intelligence(assets, p_summary, request.risk_level, risk_data, base_curr, rsi_cache=_rsi_cache)
    apex_advisor = _run_apex_advisor(assets, full_prices, ohlc_data, p_summary, live_vix=_vix_live)

    # ── Notification Collection (V6.0: consolidated prioritized list) ──
    priority_directives = []

    try:
        for alert in alpha_alerts:
            if alert.get("conviction", 0) > 88:
                priority_directives.append({
                    "priority_val": 0,
                    "priority": "CRITICAL",
                    "action": f"ALPHA: {alert['symbol']} — {alert['title']}",
                    "detail": f"Entry £{alert.get('entry_price', 0):,.2f} | Conviction {alert['conviction']}%",
                    "rationale": alert['rationale']
                })
    except Exception as e: print(f"[Telegram] Alpha alerts collect error: {e}")

    try:
        for tk, logic in apex_advisor.get("scores", {}).items():
            if logic.get("score", 0) > 85:
                priority_directives.append({
                    "priority_val": 1,
                    "priority": "HIGH",
                    "action": f"ADVISOR: {tk} — {logic['action']}",
                    "detail": f"Strategy: {logic['strat']} | Score: {logic['score']}/100",
                    "rationale": logic['why']
                })
    except Exception as e: print(f"[Telegram] Advisor collect error: {e}")

    # ── CIO LLM — live TAA (falls back to rule engine when disabled/error) ─
    ideal_bp = advanced.get("ideal_blueprint", {})
    if request.enable_cio_llm:
        cio = await _cio_with_fallback(
            holdings, assets, p_summary, ideal_bp, 
            alpha_alerts=alpha_alerts, 
            apex_advisor=apex_advisor, 
            advanced_intel=advanced,
            rsi_cache=_rsi_cache, 
            live_vix=_vix_live,
            model_provider=request.model_provider
        )
    else:
        desk = _generate_tactical_desk(holdings, assets, p_summary, ohlc_data=ohlc_data, live_vix=_vix_live)
        
        # Neutral Master Verdict for non-LLM mode
        v_regime = "STABLE" if (_vix_live or 15) < 22 else "HIGH"
        mv = {
            "summary": f"VIX regime is {v_regime}. Portfolio remains aligned with strategic mandate parameters.",
            "directives": [
                {
                    "priority": "ROUTINE",
                    "action": "Maintain strategic target weights.",
                    "timeframe": "Strategic (1-2 weeks)",
                    "rationale": "No significant macro dislocations detected by the rule engine. Continue following your chosen risk mandate."
                }
            ]
        }
        if total_cash_val > 50:
            mv["directives"].insert(0, {
                "priority": "MEDIUM",
                "action": f"Review £{total_cash_val:,.0f} uninvested cash.",
                "timeframe": "Tactical (24-72h)",
                "rationale": "Significant idle cash detected. Consider deploying into your Core bucket to avoid long-term inflationary drag."
            })

        cio  = {"tactical_desk": desk,
                "tactical_blueprint": _compute_tactical_blueprint(ideal_bp, desk),
                "master_verdict": mv,
                "sentiment_updates": {}, "_llm_powered": False}
    
    tactical_desk      = cio["tactical_desk"]
    tactical_blueprint = cio["tactical_blueprint"]

    # ── Master Directive Collection ──────────────────────────────────────
    try:
        mv_data = cio.get("master_verdict", {})
        for d in mv_data.get("directives", []):
            p = d.get("priority", "").upper()
            if p in ["CRITICAL", "HIGH", "MEDIUM"]:
                priority_directives.append({
                    "priority_val": 0 if p == "CRITICAL" else 1 if p == "HIGH" else 2,
                    "priority": p,
                    "action": d["action"],
                    "detail": f"Timeframe: {d['timeframe']}",
                    "rationale": d["rationale"]
                })
    except Exception as e: print(f"[Telegram] Master verdict collect error: {e}")

    # ── SEND CONSOLIDATED ALERT ──────────────────────────────────────────
    if priority_directives:
        # Sort by Priority Value (0 is highest)
        priority_directives.sort(key=lambda x: x["priority_val"])
        
        msg = "<b>📢 VESPER CONSOLIDATED INTELLIGENCE</b>\n\n"
        for i, d in enumerate(priority_directives[:10]):
            icon = "🚨" if d["priority"] == "CRITICAL" else "⚠️" if d["priority"] == "HIGH" else "ℹ️"
            msg += f"{i+1}. {icon} <b>{d['action']}</b>\n"
            msg += f"   <i>{d['detail']}</i>\n"
            msg += f"   {d['rationale']}\n\n"
        
        msg += f"—\nAnalysis completed at {datetime.datetime.now().strftime('%H:%M:%S')}."
        await send_telegram_alert(msg)

    # ── Recompute blueprint actions using TACTICAL targets ────────────────
    _tac_tgt = tactical_blueprint.get("target", {})
    if _tac_tgt:
        _cur  = advanced["ideal_blueprint"]["current"]
        _segs = advanced["ideal_blueprint"]["segments"]
        _tac_actions = []
        for _cat, _tgt in _tac_tgt.items():
            if not isinstance(_tgt, (int, float)):
                continue
            _diff = _tgt - _cur.get(_cat, 0.0)
            if abs(_diff) > 0.05:
                _tac_actions.append({
                    "category": _cat,
                    "action":   "Increase" if _diff > 0 else "Trim",
                    "impact":   f"{abs(_diff)*100:.1f}% shift required",
                    "tickers":  _segs.get(_cat, []),
                })
        advanced["ideal_blueprint"]["strategic_target"] = dict(advanced["ideal_blueprint"]["target"])
        advanced["ideal_blueprint"]["target"]  = _tac_tgt
        advanced["ideal_blueprint"]["actions"] = _tac_actions

    for tk, upd in cio.get("sentiment_updates", {}).items():
        if tk in assets and isinstance(upd, dict):
            s = assets[tk].get("sentiment", {})
            s.update({k: v for k, v in upd.items() if v is not None})
            rsi = float(s.get("rsi") or _rsi_cache.get(tk) or 50.0)
            s["rsi_signal"] = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            assets[tk]["sentiment"] = s

    if "summary_hints" not in cio:
        hedge_eff   = advanced.get("hedge_optimizer", {}).get("hedge_efficiency", "HIGH")
        beta_val    = p_summary.get("portfolio_beta", 1.0)
        sharpe_val  = p_summary.get("sharpe_ratio", 0.5)
        cagr_val    = float(advanced.get("real_cagr", 0.05))
        drag        = sorted(assets.items(), key=lambda x: -(x[1].get("risk_contribution_pct", 0)))[:2]
        drag_str    = ", ".join(tk for tk, _ in drag) if drag else "high-vol positions"

        cio["summary_hints"] = {
            "var": ("↑ Elevated — defensive hedge is sufficient" if hedge_eff == "SUFFICIENT" else "↑ Elevated — add SGLN.L gold hedge"),
            "beta": ("✓ Neutral market exposure" if 0.8 <= beta_val <= 1.1 else f"↑ High — rotate partial {max(assets, key=lambda t: assets[t]['weight'], default='growth')} to VIG/SGLN.L" if beta_val > 1.1 else "↓ Low — consider adding QQQ/SPY for growth"),
            "sharpe": ("✓ Efficient alpha capture" if sharpe_val > 0.8 else f"↓ Cut drag: {drag_str}" if sharpe_val < 0.4 else "↓ Noisy — add Quality Factor"),
            "cagr": ("✓ Beating inflation — maintain growth tilt" if cagr_val > 0.05 else "↑ Marginal — increase growth allocation" if cagr_val > 0.02 else "↑ Critical — restructure to CMA-aligned weights"),
        }

    if "quant_intelligence" not in cio:
        dr = float(advanced.get("attribution", {}).get("div_ratio", 1.0))
        if dr >= 1.3: cio["quant_intelligence"] = f"Structural diversification is operational. DR {dr:.2f} confirms factor independence."
        elif dr >= 1.1: cio["quant_intelligence"] = f"Moderate diversification detected. DR {dr:.2f} shows partial factor independence."
        else: cio["quant_intelligence"] = f"Diversification is failing. DR {dr:.2f} indicates high factor overlap."

    advanced["summary_hints"] = cio.get("summary_hints", {})
    market_outlook = _get_market_outlook(assets, p_summary)
    market_outlook["quant_intelligence"] = cio.get("quant_intelligence", "")

    llm_outlook = cio.get("market_outlook")
    if isinstance(llm_outlook, list) and len(llm_outlook) >= 3:
        market_outlook["points"] = llm_outlook
        market_outlook["_source"] = "llm"
    if cio.get("_llm_error"): market_outlook["_llm_error"] = cio["_llm_error"]
    if cio.get("strategic_commentary"): market_outlook["commentary"] = cio["strategic_commentary"]
    if cio.get("future_outlook"): market_outlook["future_outlook"] = cio["future_outlook"]

    return {
        "generated_at": datetime.datetime.now().isoformat(),
        "master_verdict": cio.get("master_verdict", {}),
        "portfolio": p_summary, "assets": assets, "risk": risk_data, "advanced_intel": advanced,
        "alpha_alerts": alpha_alerts,
        "recommendations": _generate_recommendations(assets, p_summary, request.risk_level, base_curr),
        "events": {tk: _extract_events(infos.get(tk, {})) for tk in tickers},
        "global_macro_events": _get_global_events(),
        "market_outlook": market_outlook,
        "tactical_desk": tactical_desk,
        "tactical_blueprint": tactical_blueprint,
        "apex_advisor": apex_advisor,
    }


@app.get("/api/quotes")
async def get_quotes(tickers: str = Query(..., description="Comma-separated ticker symbols")):
    """Return price, previous close, daily change and name for each ticker.
    Used by the live ticker bar — parallel fetches, max 30 symbols."""
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()][:30]

    async def _quote(tk: str):
        try:
            api_tk = _resolve_ticker(tk)
            info = await _in_thread(lambda: yf.Ticker(api_tk).info) or {}
            price = _to_float(
                info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0
            )
            prev  = _to_float(info.get("previousClose") or info.get("regularMarketPreviousClose") or price)
            # GBX (pence) → GBP
            ccy = info.get("currency") or "USD"
            if ccy == "GBp":
                price /= 100; prev /= 100; ccy = "GBP"
            chg = price - prev
            pct = (chg / prev * 100) if prev else 0.0
            name = (info.get("shortName") or info.get("longName") or tk)[:30]
            result = {
                "ticker": tk, "name": name,
                "price": round(price, 4), "prev_close": round(prev, 4),
                "change": round(chg, 4), "change_pct": round(pct, 4),
                "currency": ccy,
            }
            # Flag SEDOL/fund tickers that Yahoo couldn't resolve
            if price == 0 and _is_sedol_ticker(tk):
                result["use_broker_price"] = True
            return result
        except Exception:
            result = {"ticker": tk, "name": tk, "price": 0, "prev_close": 0,
                    "change": 0, "change_pct": 0, "currency": "USD"}
            if _is_sedol_ticker(tk):
                result["use_broker_price"] = True
            return result

    results = await asyncio.gather(*[_quote(s) for s in syms])
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M UTC")
    return {"quotes": list(results), "timestamp": ts}


# ── P&L Desk API routes ───────────────────────────────────────────────────────

@app.post("/api/parse-transactions")
async def parse_transactions(request: TransactionRequest):
    """Parse a broker CSV, run P&L engine, optionally persist to Firestore."""
    try:
        df = _parse_ii_csv(request.csv_text, request.account_type)
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid transactions found in CSV.")

        result = _run_pnl_engine(df, request.account_type)
        response = {"by_account": {request.account_type: result, "all": result}}

        # Persist to Firestore if auth token provided
        new_count, dup_count = 0, 0
        if request.id_token and _FIREBASE_OK:
            try:
                decoded = _fb_auth.verify_id_token(request.id_token)
                uid = decoded["uid"]
                new_count, dup_count = await _in_thread(
                    _save_transactions_to_firestore, uid, request.csv_text, request.account_type
                )
                # Reload full ledger from Firestore (includes all accounts)
                full = await _in_thread(_load_user_transactions, uid)
                response = full
            except Exception as e:
                # If Firestore save fails, still return the parsed result
                pass

        response["new_count"] = new_count
        response["dup_count"] = dup_count
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/load-ledger")
async def load_ledger(request: Request):
    """Load saved transaction data from Firestore for the authenticated user."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token.")
    token = auth_header.split(" ", 1)[1]
    if not _FIREBASE_OK:
        raise HTTPException(status_code=503, detail="Firebase not configured.")
    try:
        decoded = _fb_auth.verify_id_token(token)
        uid = decoded["uid"]
        result = await _in_thread(_load_user_transactions, uid)
        return result
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {e}")


@app.delete("/api/clear-ledger")
async def clear_ledger(request: Request, account_type: str = ""):
    """Delete saved transaction data from Firestore."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token.")
    token = auth_header.split(" ", 1)[1]
    if not _FIREBASE_OK:
        raise HTTPException(status_code=503, detail="Firebase not configured.")
    try:
        decoded = _fb_auth.verify_id_token(token)
        uid = decoded["uid"]
        coll = _FS_CLIENT.collection("users").document(uid).collection("transactions")
        deleted = 0
        for doc in coll.stream():
            if account_type and doc.to_dict().get("account_type") != account_type:
                continue
            doc.reference.delete()
            deleted += 1
        return {"deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {e}")


@app.get("/health")
async def health(): return {"status": "ok"}

@app.get("/")
async def ui(): return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# ── Firebase Cloud Function wrapper ──────────────────────────────────────────
try:
    from firebase_functions import https_fn, options

    @https_fn.on_request(memory=options.MemoryOption.GB_2, timeout_sec=300, region="us-central1")
    def vesper_api(req: https_fn.Request) -> https_fn.Response:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        path = req.path
        if path.startswith("/vesper_api"): path = path[len("/vesper_api"):]
        if not path: path = "/"
        response = client.request(method=req.method, url=path, params=req.args, headers=dict(req.headers), content=req.data)
        return https_fn.Response(response.content, status=response.status_code, headers=dict(response.headers))
except ImportError:
    pass  # Not running in Firebase Cloud Functions environment
