"""
Vesper v4.0 — Firebase Cloud Intelligence Engine (Institutional Apex)
"""

import asyncio
import datetime
import time
import traceback
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from firebase_functions import https_fn, options
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")
FINBERT_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

CACHE_TTL = 3600
RISK_FREE_RATE = 0.05

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PortfolioRequest(BaseModel):
    holdings: Dict[str, float]
    risk_level: str = "Balanced"

    @field_validator("holdings")
    @classmethod
    def validate_holdings(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v: raise ValueError("Provide at least one ticker.")
        return {k.upper().strip(): float(q) for k, q in v.items()}

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────
# Quantitative Logic
# ─────────────────────────────────────────────────────────────────────

async def _fetch_full_history(tickers: List[str]) -> pd.DataFrame:
    all_series = {}
    async def fetch_one(ticker: str):
        try:
            t = yf.Ticker(ticker)
            h = await asyncio.to_thread(lambda: t.history(period="max", auto_adjust=True))
            if not h.empty:
                if h.index.tz is not None: h.index = h.index.tz_localize(None)
                return ticker, h["Close"]
        except: pass
        return ticker, pd.Series()
    results = await asyncio.gather(*[fetch_one(tk) for tk in tickers])
    for tk, s in results:
        if not s.empty: all_series[tk] = s
    return pd.DataFrame(all_series)

def _compute_correlation_from_prices(prices: pd.DataFrame, tickers: List[str]) -> Dict[str, Any]:
    try:
        if prices.empty: return {"tickers": tickers, "matrix": [], "error": "No data"}
        p_1y = prices[prices.index >= (prices.index[-1] - pd.Timedelta(days=365.25))]
        if len(p_1y) < 5: return {"tickers": tickers, "matrix": []}
        corr_df = p_1y.pct_change().dropna().corr()
        valid = [t for t in tickers if t in corr_df.columns]
        matrix = [[round(float(corr_df.loc[a, b]), 4) if np.isfinite(corr_df.loc[a, b]) else 0 for b in valid] for a in valid]
        return {"tickers": valid, "matrix": matrix}
    except: return {"tickers": tickers, "matrix": []}

def _compute_risk_history_from_prices(prices: pd.DataFrame, tickers: List[str], weights_dict: Dict[str, float]) -> Dict[str, Any]:
    empty = {"asset_risk": {t: {} for t in tickers}, "portfolio_risk": {}, "future_value": {}, "history": {}, "risk_contributions": {}, "asset_vols": {}, "asset_betas": {}}
    try:
        if prices.empty: return empty
        asset_risk = {}
        history = {}
        asset_vols = {}
        
        def calc_cagr(s, y):
            if y <= 0: return None
            try:
                s = s.dropna(); s = s[s > 0]
                if len(s) < 2: return None
                val = (s.iloc[-1] / s.iloc[0])**(1/y)-1
                return round(float(val), 4) if np.isfinite(val) else None
            except: return None

        for t in tickers:
            if t not in prices.columns: continue
            p = prices[t].dropna(); 
            if len(p) < 2: continue
            last = p.index[-1]
            p_1y = p[p.index >= (last - pd.Timedelta(days=365.25))]
            if not p_1y.empty:
                normed = (p_1y / p_1y.iloc[0] * 100).round(2)
                history[t] = {"dates": [d.strftime("%Y-%m-%d") for d in normed.index[::5]], "values": normed.values[::5].tolist()}
            
            vol = round(float(p_1y.pct_change().std()*np.sqrt(252)), 4) if len(p_1y) > 5 else 0.20
            asset_vols[t] = vol
            
            years_total = max((p.index[-1]-p.index[0]).days/365.25, 0.1)
            asset_risk[t] = {
                "inception_date": p.index[0].strftime("%Y-%m-%d"),
                "cagr_inception": calc_cagr(p, years_total),
                "return_1y": round(float(p_1y.iloc[-1]/p_1y.iloc[0]-1), 4) if len(p_1y) > 1 else "N/A",
                "volatility": vol
            }

        filled = prices.ffill().dropna()
        valid = [t for t in tickers if t in filled.columns]
        if not valid: return empty
        w = np.array([weights_dict.get(t, 0.0) for t in valid]); 
        if w.sum() > 0: w = w/w.sum() 
        rets = filled[valid].pct_change().dropna(); 
        if rets.empty: return empty
        
        # Risk Contributions (v4.0)
        cov_matrix = rets.cov() * 252
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        port_vol = np.sqrt(port_var)
        marginal_contribution = np.dot(cov_matrix, w) / port_vol
        component_contribution = w * marginal_contribution
        risk_contributions = {tk: round(float(component_contribution[i] / port_vol), 4) for i, tk in enumerate(valid)}
        
        # Beta (v4.0)
        asset_betas = {}
        if "SPY" in filled.columns:
            spy_rets = filled["SPY"].pct_change().dropna()
            for tk in valid:
                try:
                    c = np.cov(rets[tk], spy_rets.reindex(rets.index))[0,1]
                    v = np.var(spy_rets)
                    asset_betas[tk] = round(float(c/v), 3) if v != 0 else 1.0
                except: asset_betas[tk] = 1.0

        port_rets = rets.values @ w
        port_series = pd.Series(port_rets, index=rets.index); port_cum = (1+port_series).cumprod()
        ann_vol = port_series[port_series.index >= (filled.index[-1]-pd.Timedelta(days=365.25))].std()*np.sqrt(252)
        cagr_p = calc_cagr(port_cum, max(len(port_series)/252, 0.1)) or 0
        sharpe = round((cagr_p - RISK_FREE_RATE)/ann_vol, 2) if ann_vol > 0 else 0
        var_95_pct = round(float(1.645 * (ann_vol / np.sqrt(252))), 4)
        
        # Projections
        n_sims, n_years, n_days = 2000, 10, 252
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(port_rets), size=(n_sims, n_years*n_days))
        sampled = port_rets[idx].reshape(n_sims, n_years, n_days)
        paths = np.column_stack([np.ones(n_sims), np.cumprod(np.prod(1+sampled, axis=2), axis=1)])
        future_value = {"median_cagr": round(float(np.percentile(paths[:,-1], 50)**(1/10)-1), 4), "p10": np.percentile(paths, 10, axis=0).tolist(), "p50": np.percentile(paths, 50, axis=0).tolist(), "p90": np.percentile(paths, 90, axis=0).tolist()}
        
        portfolio_risk = {"total_value_basis": 100, "volatility": round(float(ann_vol), 4), "sharpe_ratio": sharpe, "var_95_daily": var_95_pct}
        return {"asset_risk": asset_risk, "portfolio_risk": portfolio_risk, "future_value": future_value, "history": history, "risk_contributions": risk_contributions, "asset_vols": asset_vols, "asset_betas": asset_betas}
    except: return empty

def _get_advanced_intelligence(assets, portfolio, risk_level, risk_data):
    total_val = portfolio.get("total_value", 0)
    beta = portfolio.get("portfolio_beta", 1.0)
    var_95 = portfolio.get("var_95_daily", 0)
    risk_contributions = risk_data.get("risk_contributions", {})
    asset_vols = risk_data.get("asset_vols", {})
    asset_betas = risk_data.get("asset_betas", {})
    
    # 1. Kill Switch Logic
    guardrail_breached = var_95 > 0.035 or beta > 1.3
    kill_switch_alert = "SYSTEM OVERLOAD: Quantitative guardrails breached. Immediate rotation to defensive proxy (GLD/VIG) required." if guardrail_breached else None

    # 2. Risk Contribution Penalty
    high_risk_asset = next((tk for tk, c in risk_contributions.items() if c > 0.6), None)
    div_penalty = round((risk_contributions[high_risk_asset] - 0.6) * 100, 2) if high_risk_asset else 0.0

    # 3. Diversification Alpha
    weighted_vol_sum = sum(assets[tk]["weight"] * asset_vols.get(tk, 0.2) for tk in assets)
    port_vol = portfolio.get("volatility", 0.15)
    div_ratio = round(weighted_vol_sum / port_vol, 3) if port_vol > 0 else 1.0

    # 4. FX Sensitivity (Cable)
    usd_val = sum(a["value"] for a in assets.values() if a["currency"] == "USD")
    fx_impact_5pct = round(usd_val * 0.05, 2)

    # 5. Rebalance Impact
    targets = {"Conservative": 0.20, "Balanced": 0.40, "Aggressive": 0.70}
    target_growth_w = targets.get(risk_level, 0.40)
    growth_tks = [tk for tk, a in assets.items() if a["sector"] in ("Technology", "Communication Services") or (_to_float(a.get("pe_ratio")) or 0) > 30]
    
    proj_weights = {}
    if growth_tks:
        g_w = target_growth_w / len(growth_tks)
        others = [tk for tk in assets if tk not in growth_tks]
        o_w = (1.0 - target_growth_w) / len(others) if others else 0
        for tk in assets: proj_weights[tk] = g_w if tk in growth_tks else o_w
    else:
        proj_weights = {tk: 1.0/len(assets) for tk in assets}

    proj_beta = round(sum(proj_weights[tk] * asset_betas.get(tk, 1.0) for tk in assets), 2)
    proj_sharpe = round(portfolio.get("sharpe_ratio", 0.5) * (target_growth_w / max(0.01, sum(assets[tk]["weight"] for tk in growth_tks))), 2)

    for tk, a in assets.items():
        s = a.get("sentiment", {})
        curr_score = s.get("bullish", 0.5)
        s["sentiment_delta"] = round(curr_score - 0.55, 3) 
        s["exhaustion_alert"] = s["sentiment_delta"] < -0.15

    nasdaq_impact = round(total_val * beta * -0.10, 2)
    current_growth_w = sum(assets[tk]["weight"] for tk in growth_tks)
    drift = current_growth_w - target_growth_w
    trade_suggestion = None
    if abs(drift) > 0.05:
        action = "Sell" if drift > 0 else "Buy"
        amt = round(abs(drift) * total_val, 2)
        source = max(growth_tks, key=lambda x: assets[x]["weight"]) if drift > 0 and growth_tks else "Growth Proxy"
        trade_suggestion = f"{action} ${amt:,.0f} of Growth ({source}) to re-align."
    
    top_2_weight = sum(sorted([a["weight"] for a in assets.values()], reverse=True)[:2])
    outlook = sum((2 if a.get("sentiment", {}).get("label") == "positive" else -2 if a.get("sentiment", {}).get("label") == "negative" else 0) for a in assets.values())
    div_safety = {tk: max(0, min(100, round(100 - (_to_float(a.get("pe_ratio")) or 20)*1.2 - (_to_float(a.get("dividend_yield")) or 0)*400))) for tk, a in assets.items()}
    
    return {
        "scenarios": {"nasdaq_10": nasdaq_impact, "fx_5pct_shock": fx_impact_5pct}, 
        "rebalancing": {"current": round(current_growth_w, 4), "drift": round(drift, 4), "trade": trade_suggestion, "projected_beta": proj_beta, "projected_sharpe": proj_sharpe}, 
        "attribution": {"factor": "Growth" if current_growth_w > 0.5 else "Value", "concentration": "CRITICAL" if top_2_weight > 0.7 else "HEALTHY", "div_ratio": div_ratio}, 
        "outlook_score": max(-10, min(10, outlook)), 
        "div_safety": div_safety,
        "diversification_penalty": div_penalty,
        "kill_switch": kill_switch_alert,
        "fx_exposure_risk": "HIGH" if usd_val / total_val > 0.7 else "LOW"
    }

async def _get_sentiment(ticker: str) -> Dict[str, Any]:
    PROXY_MAP = {"VWRP": ["Global Economy", "FTSE All-World"], "SPY": ["S&P 500", "US Economy"], "QQQ": ["Nasdaq 100", "Tech Stocks"]}
    async def fetch_headlines(tk):
        try:
            t = yf.Ticker(tk)
            news = await asyncio.to_thread(lambda: t.news)
            return [n.get("content", {}).get("title") for n in news[:10] if n.get("content", {}).get("title")]
        except: return []
    headlines = await fetch_headlines(ticker)
    if not headlines and "." in ticker: headlines = await fetch_headlines(ticker.split(".")[0])
    clean_tk = ticker.split(".")[0].upper()
    if clean_tk in PROXY_MAP:
        for proxy in PROXY_MAP[clean_tk]: headlines.extend((await fetch_headlines(proxy))[:3])
    if not headlines: return {"bullish": 0, "bearish": 0, "neutral": 1, "label": "no signals", "headline_count": 0, "headlines": []}
    headlines = list(dict.fromkeys(headlines))[:20]
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(FINBERT_API_URL, headers=headers, json={"inputs": headlines}, timeout=30.0)
            results = resp.json()
            total_b = total_r = total_n = 0.0
            for res in results:
                scores = res if isinstance(res, list) else [res]
                for s in scores:
                    if s.get("label") == "positive": total_b += s.get("score", 0)
                    elif s.get("label") == "negative": total_r += s.get("score", 0)
                    else: total_n += s.get("score", 0)
            avg_b, avg_r = round(total_b/len(headlines), 3), round(total_r/len(headlines), 3)
            return {"bullish": avg_b, "bearish": avg_r, "neutral": round(total_n/len(headlines), 3), "label": "positive" if avg_b > avg_r and avg_b > 0.35 else "negative" if avg_r > avg_b and avg_r > 0.35 else "neutral", "headline_count": len(headlines), "headlines": headlines}
    except: return {"bullish": 0, "bearish": 0, "neutral": 1, "label": "inference failed", "headline_count": len(headlines), "headlines": headlines}

@app.post("/api/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    try:
        holdings = request.holdings; tickers = list(holdings.keys())
        async def _fetch_info(ticker):
            t = yf.Ticker(ticker); info = await asyncio.to_thread(lambda: t.info) or {}
            p = _to_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice") or info.get("previousClose"))
            if not p and "." not in ticker:
                alt = yf.Ticker(f"{ticker}.L"); ai = await asyncio.to_thread(lambda: alt.info)
                if ai and _to_float(ai.get("regularMarketPrice")): return f"{ticker}.L", ai
            return ticker, info
        res = await asyncio.gather(*[_fetch_info(t) for t in tickers])
        infos = {tk: info for tk, info in res}; resolved = {tk: holdings[tickers[i]] for i, (tk, info) in enumerate(res)}
        tickers = list(resolved.keys()); holdings = resolved; assets = {}
        for tk in tickers:
            info = infos.get(tk, {}); p = _to_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice") or 0); curr = safe_fetch(info, "currency", "USD")
            if curr in ("GBp", "GBX"): p /= 100.0; curr = "GBP"
            inst_flow = _to_float(info.get("heldPercentInstitutions"))
            assets[tk] = {"name": safe_fetch(info, "longName", tk), "price": round(p, 2), "quantity": holdings[tk], "value": round(p*holdings[tk], 2), "dividend_yield": _round(info.get("dividendYield")), "expense_ratio": _round(info.get("expenseRatio")), "pe_ratio": _round(info.get("trailingPE")), "beta": _round(info.get("beta")), "sector": safe_fetch(info, "sector", safe_fetch(info, "category", "N/A")), "currency": curr, "institutional_flow_score": round(inst_flow*100, 1) if inst_flow else 50.0}
        total_val = sum(a["value"] for a in assets.values())
        if total_val == 0: raise HTTPException(422, "No prices found")
        for a in assets.values(): a["weight"] = round(a["value"]/total_val, 4)
        full_prices = await _fetch_full_history(list(set(tickers + ["SPY"])))
        risk_data = _compute_risk_history_from_prices(full_prices, tickers, {tk: assets[tk]["weight"] for tk in tickers})
        sentiments = await asyncio.gather(*[_get_sentiment(t) for t in tickers])
        for tk, s in zip(tickers, sentiments): 
            assets[tk]["sentiment"] = s
            assets[tk]["risk_contribution_pct"] = round(risk_data.get("risk_contributions", {}).get(tk, 0) * 100, 2)
        p_summary = {"total_value": round(total_val, 2), "weighted_dividend_yield": round(sum((_to_float(a["dividend_yield"]) or 0)*a["weight"] for a in assets.values()), 4), "portfolio_beta": round(sum((_to_float(a["beta"]) or 1)*a["weight"] for a in assets.values()), 2), "asset_count": len(tickers), **risk_data.get("portfolio_risk", {})}
        
        advanced = _get_advanced_intelligence(assets, p_summary, request.risk_level, risk_data)
        
        return {"portfolio": p_summary, "assets": assets, "risk": risk_data, "advanced_intel": advanced}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "cloud_api_active": bool(HF_TOKEN)}

@https_fn.on_request(memory=options.MemoryOption.GB_2, timeout_sec=300, region="us-central1")
def vesper_api(req: https_fn.Request) -> https_fn.Response:
    from fastapi.testclient import TestClient
    global HF_TOKEN
    HF_TOKEN = os.getenv("HF_TOKEN")
    client = TestClient(app)
    path = req.path
    if path.startswith("/vesper_api"): path = path[len("/vesper_api"):]
    if not path: path = "/"
    response = client.request(method=req.method, url=path, params=req.args, headers=dict(req.headers), content=req.data)
    return https_fn.Response(response.content, status=response.status_code, headers=dict(response.headers))
