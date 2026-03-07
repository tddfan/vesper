"""
Vesper v5.0 — Intelligence Engine (Multi-Currency Apex)
"""

from __future__ import annotations

import asyncio
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
2. The latest global macroeconomic and geopolitical news headlines.

YOUR TASK:
Analyse the news and generate 2–3 short-term Tactical Asset Allocation (TAA) trades to hedge \
incoming risks or exploit immediate market dislocations. Also adjust the broader "Tactical \
Blueprint" weights (Growth, Core, Defensive) based on the current macro regime.

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
  "tactical_desk": [
    {
      "icon": "🛢️",
      "severity": "GEO-RISK",
      "time_horizon": "72h",
      "event": "Short headline summarising the risk",
      "action": "Buy",
      "ticker": "NRGG.L",
      "amount": 5000,
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

ALSO include these two additional top-level keys in the JSON:

"summary_hints": {
  "var":    "One concise line (≤12 words). Portfolio-aware VaR advice referencing actual defensive holdings.",
  "beta":   "One concise line (≤12 words). Beta advice referencing actual tickers held.",
  "sharpe": "One concise line (≤12 words). Sharpe/efficiency advice referencing actual drag assets or quality.",
  "cagr":   "One concise line (≤12 words). Real return advice referencing the CMA-blended growth outlook."
},
"quant_intelligence": "2–3 sentences. Institutional-grade diversification commentary. Mention the specific DR value, reference 1–2 actual tickers held, and give a precise actionable insight."

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
}

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
    enable_cio_llm: bool = True            # False → skip LLM, use rule engine only

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
    "fee transfer", "total monthly fee",
    "bed & isa transfer",    # GIA→ISA Bed & ISA (debit side)
]
# Descriptions starting with "PAYMENT" followed by reference → TRANSFER_IN
# e.g. "PAYMENT Q5724180656JMR S SHARMA"

def _tax_year(dt: datetime.date) -> str:
    """UK tax year: 6-Apr to 5-Apr.  2025-03-15 → '2024/25', 2025-04-06 → '2025/26'."""
    y, m, d = dt.year, dt.month, dt.day
    if m > 4 or (m == 4 and d >= 6):
        return f"{y}/{str((y + 1) % 100).zfill(2)}"
    return f"{y - 1}/{str(y % 100).zfill(2)}"

def _parse_gbp(raw: str) -> Optional[float]:
    """Parse '£1,234.56', '(£1,234.56)', 'n/a' → float or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "n/a":
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = re.sub(r"[£,\s\ufeff]", "", s)
    try:
        v = float(s)
        return -v if neg else v
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
            inst = sym.upper()
        elif sedol:
            inst = f"SEDOL:{sedol.upper()}"

        qty_raw = (raw.get("Quantity") or "").strip()
        qty = None
        if qty_raw and qty_raw.lower() != "n/a":
            try:
                qty = float(qty_raw.replace(",", ""))
            except ValueError:
                qty = None

        price = _parse_gbp(raw.get("Price"))
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
    transfer_window_end: Optional[datetime.date] = None

    total_personal_contribution = 0.0
    cg_by_year: Dict[str, float] = {}
    div_by_year: Dict[str, float] = {}
    int_by_year: Dict[str, float] = {}
    yearly: Dict[str, Dict] = {}         # tax_year → accumulators

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
        _ensure_year(ty)
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

        elif tx == "DIVIDEND":
            div_by_year[ty] = round(div_by_year.get(ty, 0) + credit, 2)
            yearly[ty]["dividends"] += credit
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

        elif tx == "TRANSFER_OUT":
            transfer_out_detail.append({
                "date": date_str, "tax_year": ty,
                "symbol": inst or "CASH",
                "amount": round(debit, 2), "type": "Transfer Out",
                "description": desc,
            })

    # ── 4. Build open positions ──────────────────────────────────────────────
    open_positions = {}
    total_book_cost = 0
    for inst, pool in pools.items():
        if pool["qty"] > 0.001:
            avg = round(pool["total_cost"] / pool["qty"], 4)
            tc = round(pool["total_cost"], 2)
            open_positions[inst] = {
                "qty": round(pool["qty"], 4),
                "avg_cost": avg,
                "total_cost": tc,
            }
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
                merged_pos[sym] = {"qty": 0, "cost": 0}
            merged_pos[sym]["qty"] += p["qty"]
            merged_pos[sym]["cost"] += p["total_cost"]

    open_positions = {}
    for sym, p in merged_pos.items():
        if p["qty"] > 0.001:
            open_positions[sym] = {
                "qty": round(p["qty"], 4),
                "avg_cost": round(p["cost"] / p["qty"], 4),
                "total_cost": round(p["cost"], 2),
            }

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

async def _fetch_full_history(tickers: List[str]) -> pd.DataFrame:
    all_series = {}
    async def fetch_one(ticker: str):
        try:
            t = yf.Ticker(ticker)
            h = await _in_thread(lambda: t.history(period="max", auto_adjust=True))
            if not h.empty:
                if h.index.tz is not None: h.index = h.index.tz_localize(None)
                return ticker, h["Close"]
        except: pass
        return ticker, pd.Series()
    results = await asyncio.gather(*[fetch_one(tk) for tk in tickers])
    for tk, s in results:
        if not s.empty: all_series[tk] = s
    return pd.DataFrame(all_series)

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

def _get_advanced_intelligence(assets, portfolio, risk_level, risk_data, base_currency):
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
            a["weight"] * _UNDERLYING_FX.get(tk.split('.')[0].upper(),
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
        tk.split('.')[0].upper() in _GROWTH_BASES or
        a["sector"] in ("Technology", "Communication Services") or
        (_to_float(a.get("pe_ratio")) or 0) > 30]
    defensive_tks = [tk for tk, a in assets.items() if
        tk.split('.')[0].upper() in _DEFENSIVE_BASES or
        "Treasury" in a["name"] or "Gold" in a["name"] or "Bond" in a["name"] or
        tk in ("TN28.L",)]
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
        _rsi  = _mock_rsi(_tk)                                          # 20–80
        _beta = _to_float(_a.get("beta")) or 1.0
        _ter  = (_to_float(_a.get("ter"))
                 or _KNOWN_TER.get(_tk.split('.')[0].upper(), 0.003))
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
    existing_gold_w = sum(a["weight"] for tk, a in assets.items() if tk.split('.')[0].upper() in _GOLD_BASES)
    if existing_gold_w >= 0.10:
        hedge_opt = {"optimal_gld_weight": 0.0, "hedge_efficiency": "SUFFICIENT",
                     "note": f"Gold at {existing_gold_w*100:.0f}% — adequate. Consider IGLT.L for duration hedge."}
    else:
        needed = max(0.0, round(0.12 - existing_gold_w, 2))
        hedge_opt = {"optimal_gld_weight": needed, "hedge_efficiency": "HIGH" if beta > 1.1 else "LOW", "note": None}

    return {
        "scenarios": {"nasdaq_10": round(total_val*beta*-0.1, 2), "fx_5pct_shock": round(non_base_val*0.05, 2)},
        "rebalancing": {"current": round(curr_g_w, 4), "drift": round(drift, 4), "trade": trade_suggestion, "projected_beta": round(beta - (drift*0.2), 2), "projected_sharpe": round(portfolio.get("sharpe_ratio", 0.5)*1.2, 2), "transaction_cost_estimate": cost},
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
    
    return {"points": points, "commentary": " ".join(comm), "future_outlook": future_outlook}

def _get_global_events():
    now = datetime.datetime.now()
    return [
        {"d": (now + pd.Timedelta(days=12)).strftime("%Y-%m-%d"), "e": "US Fed Meeting", "i": "Rate decision."},
        {"d": (now + pd.Timedelta(days=18)).strftime("%Y-%m-%d"), "e": "US CPI Data", "i": "Inflation trigger."},
        {"d": (now + pd.Timedelta(days=25)).strftime("%Y-%m-%d"), "e": "BoE Meeting", "i": "GBP driver."}
    ]

def _generate_recommendations(assets, portfolio, risk_level, base_currency="GBP"):
    recs = []
    is_ucits = base_currency in ("GBP", "EUR") or any(tk.endswith('.L') or tk.endswith('.AS') or tk.endswith('.DE') for tk in assets)
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
    # Acc (Accumulating) funds in a GIA are still subject to UK dividend tax via
    # Excess Reportable Income (ERI). Advising clients to switch to Acc to avoid GIA
    # tax is a compliance error. Flag this if any fund name contains "Acc".
    if base_currency == "GBP":
        acc_tks = [tk for tk, a in assets.items() if "acc" in (a.get("name") or "").lower()]
        if acc_tks:
            recs.append({
                "type": "warning", "icon": "🏦",
                "title": "GIA Tax Alert: ERI on Acc Funds",
                "detail": "Acc funds in GIA still taxable via ERI.",
                "rationale": (
                    "UK HMRC taxes Excess Reportable Income (ERI) from Accumulating (Acc) funds "
                    "in a GIA as dividend income — even though no cash is distributed. "
                    "Switching to an Acc share class does NOT reduce GIA dividend or income tax. "
                    "Recommended action: Bed & ISA transfer at UK tax year start (before 5 April) "
                    f"to shelter gains inside an ISA. Affected: {', '.join(acc_tks[:3])}."
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

def _mock_rsi(ticker: str) -> float:
    """Deterministic mock RSI seeded by ticker chars. Range: 20–80."""
    seed = sum(ord(c) for c in ticker.upper()) % 61  # 0-60
    return round(20.0 + seed, 1)


# ── Macro factor signals per ETF — shown when live FinBERT headlines unavailable ─
_MACRO_FACTORS: Dict[str, List[str]] = {
    "EQQQ": [
        "AI capex supercycle sustaining Nasdaq 100 EPS growth trajectory",
        "Fed higher-for-longer path compresses long-duration growth multiples",
        "Semiconductor capacity expansion: TSMC N2 ramp accelerating through H2",
    ],
    "QQQ":  [
        "Mega-cap tech buybacks providing structural bid beneath the index",
        "AI revenue monetisation accelerating across cloud, ad, and enterprise",
        "Options market pricing elevated implied vol — momentum fragility signal",
    ],
    "VWRP": [
        "Global equity risk premium at multi-year lows — valuation headwind building",
        "USD strength creating FX drag on 55% US weight within the fund",
        "EM allocation: China property deleveraging weighing on emerging-market returns",
    ],
    "SWRD": [
        "Developed market earnings revisions diverging: US upgrade, Europe flat",
        "Yen weakness amplifying unhedged Japan allocation within MSCI World",
    ],
    "CSPX": [
        "S&P 500 concentration: top-10 holdings exceed 35% index weight",
        "US labour market resilience supporting corporate revenue outlook",
        "PCE inflation sticky — Fed pivot expectations pushed to late 2025",
    ],
    "VUSA": [
        "S&P 500 buyback yield at 1.8% providing technical floor",
        "US consumer credit stress emerging in sub-prime auto and card delinquencies",
    ],
    "IITU": [
        "Semiconductor equipment orders inflecting higher — cycle recovery on track",
        "AI inference demand creating structural tailwind for large-cap tech",
    ],
    "SGLN": [
        "Central bank gold accumulation surpassing 1,000 tonnes per annum",
        "Real yields declining from peak — structurally supportive for gold prices",
        "De-dollarisation accelerating: BRICS nations diversifying FX reserves into gold",
    ],
    "PHGP": [
        "Gold spot holding above $2,300/oz — momentum regime intact",
        "ETF outflows moderating as institutional demand absorbs retail selling pressure",
    ],
    "NRGG": [
        "Brent crude supported by OPEC+ production discipline and Red Sea disruptions",
        "US shale production growth moderating — supply tightness emerging in H2",
        "Energy sector FCF yield at 8%+ — shareholder return programmes accelerating",
    ],
    "DFNG": [
        "NATO spending mandates: European members raising defence budgets to 2% GDP",
        "Geopolitical risk premium structurally embedded in defence sector multiples",
        "Order backlogs at record levels — 5yr revenue visibility above sector average",
    ],
    "IIND": [
        "India GDP growth at 7.2% — strongest trajectory among major economies globally",
        "Nifty 50 earnings: 15% YoY growth driven by domestic consumption and IT exports",
        "RBI holding rates as CPI trends toward 4% target — policy inflection approaching",
    ],
    "EMIM": [
        "EM earnings recovery contingent on China fiscal stimulus efficacy",
        "USD strength creating persistent FX headwind across EM commodity exporters",
        "India and Indonesia offsetting China allocation drag within the fund",
    ],
    "VFEM": [
        "EM valuation discount to DM at widest since 2005 — mean-reversion potential",
        "China property sector: policy stimulus stabilising; risk of further leg down remains",
    ],
    "ISF":  [
        "FTSE 100 value discount to US peers at historic wide — rerating catalyst needed",
        "Commodity and financials skew: FTSE 100 benefits from USD strength and yield curve",
        "BoE rate cuts supportive for domestic consumer and real estate sub-sectors",
    ],
    "VAGP": [
        "Global bond duration risk elevated as inflation stays above central bank targets",
        "Investment grade credit spreads tightening — risk appetite supportive for IG bonds",
    ],
    "VGOV": [
        "UK gilt carry premium: BoE Bank Rate at 5.25% generating positive real yield",
        "UK fiscal deficit widening — gilt supply issuance risk emerging in H2",
    ],
    "IGLH": [
        "GBP-hedged USD credit: hedge cost eroding yield pickup relative to gilts",
        "IG default rates at cyclical lows — credit quality broadly intact",
    ],
    "DAGB": [
        "Global aggregate duration exposure elevated — higher-for-longer is the base case",
        "Credit quality improving; IG spreads near post-GFC tights — limited upside",
    ],
    "AINF": [
        "Infrastructure assets: inflation-linked revenues providing real return floor",
        "Energy transition: $3T/yr capex creating secular demand for infrastructure funds",
    ],
    "SPOG": [
        "Real estate valuations rebasing lower as cap rates adjust to higher discount rates",
        "Logistics and data-centre REITs outperforming — structural demand from AI build-out",
    ],
    "WLDS": [
        "Global small cap lagging large cap — risk-off positioning reducing speculative demand",
        "Small cap earnings sensitivity to credit conditions: watch US regional bank lending",
    ],
    "VJPB": [
        "Bank of Japan yield curve control exit: JGB volatility risk to global bond markets",
        "Yen carry unwind risk: USD/JPY positioning at extreme levels — reversal fragility",
    ],
    "CMOP": [
        "Commodity super-cycle thesis: supply underinvestment meeting structural EM demand",
        "China restocking cycle: PMI recovery driving base metals upside",
    ],
    "SDIP": [
        "Dividend growth stocks: quality factor outperforming in high-rate environment",
        "Payout ratios healthy — dividend coverage supported by robust FCF generation",
    ],
}

def _mock_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Heuristic sentiment derived from the same RSI seed — deterministic and varied per ticker.
    Called whenever the real FinBERT API is unavailable or returns no headlines.
    RSI > 65  → momentum regime → net bullish scores
    RSI < 35  → fear / oversold  → net bearish scores
    35–65     → neutral band     → mixed scores
    Headlines are populated from _MACRO_FACTORS for known ETFs so the frontend
    'Macro Factor Synthesis' section always has content.
    """
    rsi   = _mock_rsi(ticker)
    # Secondary spread: weighted sum of char ordinals and positions (0–99)
    spread = sum(ord(c) * (i + 1) for i, c in enumerate(ticker.upper())) % 100 / 1000.0

    if rsi > 65:          # momentum / overbought
        bull = round(0.52 + spread, 3)
        bear = round(0.20 + spread * 0.4, 3)
    elif rsi < 35:        # oversold / fear
        bull = round(0.20 + spread * 0.4, 3)
        bear = round(0.52 + spread, 3)
    else:                 # neutral band
        bull = round(0.36 + spread * 0.8, 3)
        bear = round(0.28 + spread * 0.5, 3)

    label = ("positive" if bull > bear and bull > 0.38
             else "negative" if bear > bull and bear > 0.38
             else "neutral")

    clean = ticker.split(".")[0].upper()
    macro_headlines = _MACRO_FACTORS.get(clean, [
        "Macro regime: elevated uncertainty warrants active risk monitoring",
        "Portfolio factor exposure: review correlation to broader equity beta",
    ])

    return {
        "bullish": bull, "bearish": bear, "label": label,
        "headline_count": len(macro_headlines),
        "headlines": macro_headlines,
        "sentiment_delta": round(bull - 0.45, 2),
        "exhaustion_alert": rsi > 74,
    }


_TECH_S    = {"EQQQ","CNDX","QQQ","QQQM","SMH","SOXX","AINF","DAGB","MAGS","ARKK","WTEC","LGQG","IIND"}
_GOLD_TAA  = {"SGLN","GLD","IAU","GLDM","PHAU","HMSO","IGLN","SGLP","XGLD"}
_BOND_TAA  = {"TN28","IGLT","BND","AGG","TLT","VGLT","XGSD"}
_ENERGY_TAA = {"XOM","CVX","SHEL","BP","RDSB","OXY","NRGG","XENE","IOOG"}

def _generate_tactical_desk(holdings: Dict[str, float], assets: Dict[str, Any], portfolio: Dict[str, Any]) -> List[Dict]:
    """
    Simulated LLM Catalyst Engine.
    Scores mock live news headlines against portfolio composition and
    generates 2-3 event-driven tactical trade alerts for a 72-hour horizon.
    """
    total_val = portfolio.get("total_value", 0)
    ticker_bases = {tk.split('.')[0].upper() for tk in holdings}

    has_gold   = bool(ticker_bases & _GOLD_TAA)
    has_bonds  = bool(ticker_bases & _BOND_TAA)
    has_tech   = bool(ticker_bases & _TECH_S)
    has_energy = bool(ticker_bases & _ENERGY_TAA)

    desk: List[Dict] = []

    # Signal 1 — GEO-RISK: oil supply shock → buy energy hedge if absent
    if not has_energy:
        desk.append({
            "severity": "GEO-RISK", "icon": "🛢️",
            "event": "Oil Supply Shock — Strait of Hormuz Threatened",
            "action": "Buy", "ticker": "NRGG.L",
            "amount": round(total_val * 0.03),
            "rationale": (
                "Escalating Middle East tensions threaten ~20% of global oil transit. "
                "iShares Oil & Gas Exploration & Production UCITS ETF (NRGG.L) provides direct "
                "exposure to upstream E&P companies and acts as a natural hedge against "
                "supply-driven price spikes. Enter on any dip toward the 5-day MA; "
                "exit if Brent crude retreats below key support."
            ),
            "time_horizon": "72h",
        })

    # Signal 2 — MACRO-SHOCK: hot CPI → trim highest-weight tech by 10%
    tech_tickers = [tk for tk in holdings if tk.split('.')[0].upper() in _TECH_S]
    if has_tech and tech_tickers:
        top_tech = max(tech_tickers, key=lambda tk: assets.get(tk, {}).get("value", 0))
        trim_val = round(assets.get(top_tech, {}).get("value", 0) * 0.10)
        if trim_val > 0:
            desk.append({
                "severity": "MACRO-SHOCK", "icon": "🌡️",
                "event": "US CPI Surprise — Rate Cut Bets Collapse",
                "action": "Trim", "ticker": top_tech,
                "amount": trim_val,
                "rationale": (
                    f"Hot CPI removes near-term rate relief. Long-duration growth equities ({top_tech}) "
                    "are most exposed to discount-rate re-pricing. A 10% trim reduces duration risk "
                    "and generates dry powder for redeployment into short-duration or real assets "
                    "once the Fed's reaction function becomes clearer."
                ),
                "time_horizon": "72h",
            })

    # Signal 3 — MOMENTUM: gold breakout → add gold if absent; else add short gilts
    if not has_gold:
        desk.append({
            "severity": "MOMENTUM", "icon": "🏅",
            "event": "Gold Momentum Breakout — Institutional Safe-Haven Bid",
            "action": "Buy", "ticker": "SGLN.L",
            "amount": round(total_val * 0.02),
            "rationale": (
                "Gold clearing $2,500 signals institutional safe-haven rotation. "
                "Breakouts above key resistance historically sustain 4–8% moves over 2–4 weeks. "
                "SGLN.L (Physical Gold ETC, 0.12% TER) adds portfolio convexity against macro "
                "tail risks with structurally low correlation to equities."
            ),
            "time_horizon": "72h",
        })
    elif not has_bonds:
        desk.append({
            "severity": "MACRO-SHOCK", "icon": "🏦",
            "event": "Duration Risk Rising — Add Short-Gilt Buffer",
            "action": "Buy", "ticker": "TN28.L",
            "amount": round(total_val * 0.02),
            "rationale": (
                "Re-pricing of UK rate expectations creates a tactical opportunity in short-dated Gilts. "
                "TN28.L (0–5yr Gilt, 0.07% TER) carries near-zero duration risk while offering "
                "a real yield buffer of ~4.5%, acting as a cash-plus vehicle during equity volatility."
            ),
            "time_horizon": "72h",
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
}

# ── Junk-headline patterns to strip retail SEO chaff ─────────────────────────
_NEWS_JUNK_PATTERNS = [
    "most popular", "most bought", "top stocks for",
    "stocks to watch", "best etfs to buy", "etfs to buy",
    "should you buy", "worth buying",
]

# ── 7-day cutoff (epoch seconds) ─────────────────────────────────────────────
def _news_cutoff_ts() -> float:
    return time.time() - 7 * 86_400


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


# ── Daily headline pool (seeded by date so it's stable within a day) ─────────
_HEADLINE_POOL: List[tuple] = [
    ("Federal Reserve signals higher-for-longer rate path as PCE remains elevated", "MACRO-SHOCK"),
    ("US CPI prints 3.8% YoY, above consensus 3.5%, reigniting rate-hike fears", "MACRO-SHOCK"),
    ("Brent crude surges 4.2% as Houthi attacks disrupt Red Sea shipping lanes", "GEO-RISK"),
    ("Iran threatens Strait of Hormuz closure after US sanctions escalation", "GEO-RISK"),
    ("Nasdaq 100 posts 3-month high; AI chip sector reaches historic valuation multiple", "MOMENTUM"),
    ("UK gilt yields spike 18 bp as BoE governor flags persistent services inflation", "MACRO-SHOCK"),
    ("Gold breaks $2,450/oz as central banks accelerate de-dollarisation reserves", "MOMENTUM"),
    ("Chinese PPI contracts for 20th consecutive month; deflationary spiral deepens", "SECTOR-ROTATE"),
    ("European PMI falls to 44.2, signalling contraction for 8th consecutive month", "SECTOR-ROTATE"),
    ("OPEC+ agrees surprise output cut of 500k bbl/day effective next month", "GEO-RISK"),
    ("Nvidia surges 8% after earnings beat; options market pricing 40% implied move", "MOMENTUM"),
    ("US 2-year/10-year yield curve inverts further to −62 bp; recession signal flashes", "MACRO-SHOCK"),
    ("Russia restricts fertiliser exports, raising agricultural supply-shock risk", "GEO-RISK"),
    ("Mega-cap tech layoffs accelerate: combined 35,000 jobs cut across sector", "SECTOR-ROTATE"),
    ("Dollar index at 6-month high on safe-haven flows; EM currencies under pressure", "MACRO-SHOCK"),
    ("UK Chancellor unveils fiscal austerity package; gilt rally 12 bp on debt relief", "MACRO-SHOCK"),
    ("Semiconductor supply chain disruption: Taiwan Strait tensions intensify", "GEO-RISK"),
    ("Hedge funds at highest net-short on US equities since 2022 bear market", "MOMENTUM"),
    ("WTI crude inventory draw 8.2M bbl — largest since 2021; energy sector outperforms", "GEO-RISK"),
    ("ECB holds rates, signals two further cuts in H2; European banks rally hard", "SECTOR-ROTATE"),
]

async def _call_cio_llm(
    holdings: Dict[str, float],
    assets: Dict[str, Any],
    portfolio: Dict[str, Any],
    ideal_blueprint: Dict[str, float],
) -> Dict[str, Any]:
    """
    Calls Claude Opus 4.6 (CIO persona) to generate live TAA recommendations.
    Falls back to the deterministic rule-based engine on any error.
    """
    # ── Build daily news context (stable within a calendar day) ──────────────
    today = datetime.datetime.now(datetime.timezone.utc)
    date_seed = int(today.strftime("%Y%m%d"))
    rng_news = np.random.default_rng(date_seed)
    n_headlines = int(rng_news.integers(4, 6))
    idx = rng_news.choice(len(_HEADLINE_POOL), size=n_headlines, replace=False)
    todays_headlines = [_HEADLINE_POOL[i] for i in idx]

    # ── Format portfolio context ──────────────────────────────────────────────
    total_val  = portfolio.get("total_value", 0)
    beta       = portfolio.get("portfolio_beta", 1.0)
    vol        = portfolio.get("volatility", 0.15)
    sharpe     = portfolio.get("sharpe_ratio", 0.0)
    var_95     = portfolio.get("var_95_daily", 0.02)

    growth_pct    = sum(a["weight"] for tk, a in assets.items() if tk.split(".")[0].upper() in _GROWTH_BASES) * 100
    defensive_pct = sum(a["weight"] for tk, a in assets.items() if tk.split(".")[0].upper() in _DEFENSIVE_BASES) * 100
    core_pct      = max(0.0, 100 - growth_pct - defensive_pct)

    asset_lines = []
    for tk, a in assets.items():
        s   = a.get("sentiment", {})
        rsi = s.get("rsi", _mock_rsi(tk))
        sig = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
        cls = ("Growth"    if tk.split(".")[0].upper() in _GROWTH_BASES    else
               "Defensive" if tk.split(".")[0].upper() in _DEFENSIVE_BASES else "Core")
        asset_lines.append(
            f"  {tk}: £{a['value']:,.0f} | {a['weight']*100:.1f}% weight | {cls} | "
            f"RSI {rsi:.0f} ({sig}) | Bull {s.get('bullish', 0.33):.0%} Bear {s.get('bearish', 0.33):.0%}"
        )

    headline_text = "\n".join(f"  [{tag}] {hl}" for hl, tag in todays_headlines)
    asset_text    = "\n".join(asset_lines)

    user_msg = (
        f"## Today's Date\n{today.strftime('%A, %d %B %Y')}\n\n"
        f"## Client Portfolio — Base Currency: {portfolio.get('currency', 'GBP')}\n"
        f"{asset_text}\n\n"
        f"Totals: £{total_val:,.0f} | Beta {beta:.2f} | Vol {vol*100:.1f}% | "
        f"Sharpe {sharpe:.2f} | Daily VaR95 {var_95*100:.2f}%\n\n"
        f"## Current Allocation vs Strategic Mandate\n"
        f"Growth {growth_pct:.0f}% | Core {core_pct:.0f}% | Defensive {defensive_pct:.0f}%\n"
        f"Targets → Growth {ideal_blueprint.get('Growth', 0.45)*100:.0f}% | "
        f"Core {ideal_blueprint.get('Core', 0.40)*100:.0f}% | "
        f"Defensive {ideal_blueprint.get('Defensive', 0.15)*100:.0f}%\n\n"
        f"## Live Market Intelligence\n{headline_text}\n\n"
        f"Generate your tactical analysis now. Return ONLY the JSON object."
    )

    # ── Call Claude Opus 4.6 ─────────────────────────────────────────────────
    if not _ANTHROPIC_OK or not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("anthropic SDK or ANTHROPIC_API_KEY not available")

    client = _anthropic.AsyncAnthropic()
    async with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        # Cache the static system prompt — ~90% cheaper on repeated calls (5-min TTL)
        system=[{
            "type": "text",
            "text": _CIO_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        final = await stream.get_final_message()

    # Extract first text block (thinking blocks have no .text attr by default)
    raw = next((b.text for b in final.content if hasattr(b, "text") and b.text), "")

    # Strip accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

    result = json.loads(raw)

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


async def _cio_with_fallback(
    holdings: Dict[str, float],
    assets: Dict[str, Any],
    portfolio: Dict[str, Any],
    ideal_blueprint: Dict[str, float],
) -> Dict[str, Any]:
    """Calls the LLM; falls back silently to the rule-based engine on any error."""
    try:
        return await _call_cio_llm(holdings, assets, portfolio, ideal_blueprint)
    except Exception:
        traceback.print_exc()
        desk      = _generate_tactical_desk(holdings, assets, portfolio)
        blueprint = _compute_tactical_blueprint(ideal_blueprint, desk)
        return {"tactical_desk": desk, "tactical_blueprint": blueprint,
                "sentiment_updates": {}, "_llm_powered": False}


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

@app.post("/api/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    try:
        holdings = request.holdings; tickers = list(holdings.keys())
        base_curr = request.base_currency or "GBP"
        
        async def _fetch_info(ticker):
            try:
                # Strip composite suffix (e.g. AAPL_ISA -> AAPL) for API calls
                api_ticker = ticker.split('_')[0]
                t = yf.Ticker(api_ticker); info = await _in_thread(lambda: t.info) or {}
                p = _to_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0)
                return ticker, info, p
            except: return ticker, {}, 0
        
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
                if api_ticker.endswith(".L") and p > 500:
                    p /= 100.0
            
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
                "dividend_yield": _round(info.get("dividendYield")), 
                "pe_ratio": _round(info.get("trailingPE")), 
                "beta": _round(info.get("beta")), 
                "sector": safe_fetch(info, "sector", "N/A"),
                "institutional_flow_score": round(_to_float(info.get("heldPercentInstitutions", 0.5))*100, 1),
                "ter": _round(_to_float(info.get("annualReportExpenseRatio") or info.get("totalExpenseRatio"))
                              or _KNOWN_TER.get(tk.split('.')[0].upper())),
            }
        
        total_val = sum(a["value"] for a in assets.values())
        if total_val <= 0: total_val = 1.0
        for a in assets.values(): a["weight"] = round(a["value"]/total_val, 4)
        
        # Clean tickers (strip _ISA etc) for external API history fetch
        api_tickers = list(set([t.split('_')[0] for t in tickers] + ["SPY"]))
        full_prices = await _fetch_full_history(api_tickers)
        risk_data = _compute_risk_history_from_prices(full_prices, tickers, {tk: assets[tk]["weight"] for tk in tickers})
        sentiments = await asyncio.gather(*[_get_sentiment(t) for t in tickers])
        
        for tk, s in zip(tickers, sentiments):
            rsi_val = _mock_rsi(tk)
            s["rsi"] = rsi_val
            s["rsi_signal"] = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL"
            assets[tk]["sentiment"] = s
            assets[tk]["risk_contribution_pct"] = round(risk_data.get("risk_contributions", {}).get(tk, 0) * 100, 2)

        blended_ter = sum((_to_float(a.get("ter")) or _KNOWN_TER.get(tk.split('.')[0].upper(), 0.002)) * a["weight"]
                          for tk, a in assets.items())
        p_summary = {**risk_data["portfolio_risk"], "total_value": round(total_val, 2), "weighted_dividend_yield": round(sum((_to_float(a["dividend_yield"]) or 0)*a["weight"] for a in assets.values()), 4), "portfolio_beta": round(sum((_to_float(a["beta"]) or 1)*a["weight"] for a in assets.values()), 2), "asset_count": len(tickers), "currency": base_curr, "blended_ter": round(blended_ter, 4)}
        advanced = _get_advanced_intelligence(assets, p_summary, request.risk_level, risk_data, base_curr)

        # ── CIO LLM — live TAA (falls back to rule engine when disabled/error) ─
        ideal_bp = advanced.get("ideal_blueprint", {})
        if request.enable_cio_llm:
            cio = await _cio_with_fallback(holdings, assets, p_summary, ideal_bp)
        else:
            desk = _generate_tactical_desk(holdings, assets, p_summary)
            cio  = {"tactical_desk": desk,
                    "tactical_blueprint": _compute_tactical_blueprint(ideal_bp, desk),
                    "sentiment_updates": {}, "_llm_powered": False}
        tactical_desk      = cio["tactical_desk"]
        tactical_blueprint = cio["tactical_blueprint"]

        # ── Recompute blueprint actions using TACTICAL targets ────────────────
        # _get_advanced_intelligence computed actions against static strategic targets.
        # Now that the CIO has returned tactical_blueprint.target, recalculate the
        # deltas as (tactical_target - current_weight) so the Suggested Execution
        # Path reflects the CIO's live macro tilt — not the static strategic mandate.
        _tac_tgt = tactical_blueprint.get("target", {})
        if _tac_tgt:
            _cur  = advanced["ideal_blueprint"]["current"]
            _segs = advanced["ideal_blueprint"]["segments"]
            _tac_actions = []
            for _cat, _tgt in _tac_tgt.items():
                _diff = _tgt - _cur.get(_cat, 0.0)
                if abs(_diff) > 0.05:
                    _tac_actions.append({
                        "category": _cat,
                        "action":   "Increase" if _diff > 0 else "Trim",
                        "impact":   f"{abs(_diff)*100:.1f}% shift required",
                        "tickers":  _segs.get(_cat, []),
                    })
            # Preserve strategic target so the UI can draw the tilt arrow correctly
            advanced["ideal_blueprint"]["strategic_target"] = dict(advanced["ideal_blueprint"]["target"])
            advanced["ideal_blueprint"]["target"]  = _tac_tgt
            advanced["ideal_blueprint"]["actions"] = _tac_actions

        # Merge LLM sentiment updates (RSI + bull/bear) into existing asset data
        for tk, upd in cio.get("sentiment_updates", {}).items():
            if tk in assets and isinstance(upd, dict):
                s = assets[tk].get("sentiment", {})
                s.update({k: v for k, v in upd.items() if v is not None})
                rsi = float(s.get("rsi", _mock_rsi(tk)))
                s["rsi_signal"] = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
                assets[tk]["sentiment"] = s

        # ── summary_hints: use LLM output; fall back to data-driven computation ─
        if "summary_hints" not in cio:
            hedge_eff   = advanced.get("hedge_optimizer", {}).get("hedge_efficiency", "HIGH")
            beta_val    = p_summary.get("portfolio_beta", 1.0)
            sharpe_val  = p_summary.get("sharpe_ratio", 0.5)
            cagr_val    = float(advanced.get("real_cagr", 0.05))
            drag        = sorted(assets.items(), key=lambda x: -(x[1].get("risk_contribution_pct", 0)))[:2]
            drag_str    = ", ".join(tk for tk, _ in drag) if drag else "high-vol positions"

            cio["summary_hints"] = {
                "var": ("↑ Elevated — defensive hedge is sufficient"
                        if hedge_eff == "SUFFICIENT"
                        else "↑ Elevated — add SGLN.L gold hedge"),
                "beta": ("✓ Neutral market exposure"
                         if 0.8 <= beta_val <= 1.1
                         else f"↑ High — rotate partial {max(assets, key=lambda t: assets[t]['weight'], default='growth')} to VIG/SGLN.L"
                         if beta_val > 1.1
                         else "↓ Low — consider adding QQQ/SPY for growth"),
                "sharpe": ("✓ Efficient alpha capture"
                           if sharpe_val > 0.8
                           else f"↓ Cut drag: {drag_str}"
                           if sharpe_val < 0.4
                           else "↓ Noisy — add Quality Factor"),
                "cagr": ("✓ Beating inflation — maintain growth tilt"
                         if cagr_val > 0.05
                         else "↑ Marginal — increase growth allocation"
                         if cagr_val > 0.02
                         else "↑ Critical — restructure to CMA-aligned weights"),
            }

        # ── quant_intelligence: use LLM output; fall back to DR-driven narrative ─
        if "quant_intelligence" not in cio:
            dr = float(advanced.get("attribution", {}).get("div_ratio", 1.0))
            if dr >= 1.3:
                cio["quant_intelligence"] = (
                    f"Structural diversification is operational. DR {dr:.2f} confirms your assets are "
                    f"cancelling idiosyncratic noise — the hallmark of institutional construction. "
                    f"Maintain factor weights while monitoring correlation creep in risk-off regimes."
                )
            elif dr >= 1.1:
                cio["quant_intelligence"] = (
                    f"Moderate diversification detected. DR {dr:.2f} shows partial factor independence. "
                    f"Adding one low-correlation asset class (Gilts, Real Assets, or EM ex-China) "
                    f"would materially improve portfolio efficiency and reduce tail risk."
                )
            else:
                cio["quant_intelligence"] = (
                    f"Diversification is failing. DR {dr:.2f} indicates high factor overlap — assets "
                    f"respond identically to macro shocks, destroying multi-asset allocation benefits. "
                    f"Add IGLT.L (Gilts) or SGLN.L (Gold) for structural uncorrelation."
                )

        # Attach summary_hints to advanced_intel and quant_intelligence to market_outlook
        advanced["summary_hints"] = cio["summary_hints"]
        market_outlook = _get_market_outlook(assets, p_summary)
        market_outlook["quant_intelligence"] = cio["quant_intelligence"]

        return {
            "portfolio": p_summary, "assets": assets, "risk": risk_data, "advanced_intel": advanced,
            "recommendations": _generate_recommendations(assets, p_summary, request.risk_level, base_curr),
            "events": {tk: _extract_events(infos.get(tk, {})) for tk in tickers},
            "global_macro_events": _get_global_events(),
            "market_outlook": market_outlook,
            "tactical_desk": tactical_desk,
            "tactical_blueprint": tactical_blueprint,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


@app.get("/api/quotes")
async def get_quotes(tickers: str = Query(..., description="Comma-separated ticker symbols")):
    """Return price, previous close, daily change and name for each ticker.
    Used by the live ticker bar — parallel fetches, max 30 symbols."""
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()][:30]

    async def _quote(tk: str):
        try:
            info = await _in_thread(lambda: yf.Ticker(tk).info) or {}
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
            return {
                "ticker": tk, "name": name,
                "price": round(price, 4), "prev_close": round(prev, 4),
                "change": round(chg, 4), "change_pct": round(pct, 4),
                "currency": ccy,
            }
        except Exception:
            return {"ticker": tk, "name": tk, "price": 0, "prev_close": 0,
                    "change": 0, "change_pct": 0, "currency": "USD"}

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
