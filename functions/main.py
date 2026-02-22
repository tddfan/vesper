"""
Vesper v5.0 — Firebase Cloud Intelligence Engine (Institutional Apex)
"""

from __future__ import annotations

import asyncio
import datetime
import time
import traceback
import os
from concurrent.futures import ThreadPoolExecutor
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

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
FINBERT_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

_sentiment_cache: Dict[str, Dict[str, Any]] = {} 
_executor = ThreadPoolExecutor(max_workers=10)

CACHE_TTL = 3600
RISK_FREE_RATE = 0.05

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PortfolioRequest(BaseModel):
    holdings: Dict[str, float]
    risk_level: str = "Balanced"
    base_currency: str = "GBP"
    fallback_prices: Dict[str, float] = {}

    @field_validator("holdings")
    @classmethod
    def validate_holdings(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v: raise ValueError("Provide at least one ticker.")
        return {k.upper().strip(): float(q) for k, q in v.items()}

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

async def _get_fx_rate(from_curr: str, to_curr: str) -> float:
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        t = yf.Ticker(pair)
        info = await asyncio.to_thread(lambda: t.info)
        rate = info.get("regularMarketPrice") or info.get("previousClose")
        return float(rate) if rate else 1.0
    except: return 1.0

async def _fetch_full_history(tickers: List[str]) -> pd.DataFrame:
    all_series = {}
    async def fetch_one(ticker: str):
        try:
            t = yf.Ticker(ticker)
            h = await asyncio.to_thread(lambda: t.history(period="2y", auto_adjust=True))
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
        "asset_betas": {t: 1.0 for t in tickers}
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
            if t not in prices.columns: 
                asset_risk[t] = {"cagr_inception": 0, "volatility": 0.2}; asset_vols[t] = 0.2; continue
            p = prices[t].dropna()
            if len(p) < 2: 
                asset_risk[t] = {"cagr_inception": 0, "volatility": 0.2}; asset_vols[t] = 0.2; continue
            last = p.index[-1]; p_1y = p[p.index >= (last - pd.Timedelta(days=365.25))]
            if not p_1y.empty:
                normed = (p_1y / p_1y.iloc[0] * 100).round(2)
                history[t] = {"dates": [d.strftime("%Y-%m-%d") for d in normed.index[::5]], "values": normed.values[::5].tolist()}
            vol = round(float(p_1y.pct_change().std()*np.sqrt(252)), 4) if len(p_1y) > 5 else 0.20
            asset_vols[t] = vol; years_total = max((p.index[-1]-p.index[0]).days/365.25, 0.1)
            asset_risk[t] = {"cagr_inception": calc_cagr(p, years_total), "volatility": vol}
        filled = prices.ffill().dropna(); valid = [t for t in tickers if t in filled.columns]
        if not valid: return empty
        w = np.array([weights_dict.get(t, 0.0) for t in valid])
        if w.sum() > 0: w = w/w.sum() 
        else: w = np.ones(len(valid))/len(valid)
        rets = filled[valid].pct_change().dropna()
        if rets.empty: return empty
        cov_matrix = rets.cov() * 252; port_var = np.dot(w.T, np.dot(cov_matrix, w)); port_vol = np.sqrt(max(port_var, 1e-9))
        marginal_contribution = np.dot(cov_matrix, w) / port_vol
        risk_contributions = {tk: round(float((w[i] * marginal_contribution[i]) / port_vol), 4) for i, tk in enumerate(valid)}
        asset_betas = {}
        if "SPY" in filled.columns:
            spy_rets = filled["SPY"].pct_change().dropna()
            for tk in valid:
                try:
                    common_idx = rets[tk].index.intersection(spy_rets.index)
                    if len(common_idx) > 10:
                        c = np.cov(rets[tk].loc[common_idx], spy_rets.loc[common_idx])[0,1]
                        v = np.var(spy_rets.loc[common_idx]); asset_betas[tk] = round(float(c/v), 3) if v != 0 else 1.0
                    else: asset_betas[tk] = 1.0
                except: asset_betas[tk] = 1.0
        else: asset_betas = {t: 1.0 for t in tickers}
        port_rets = rets.values @ w; port_series = pd.Series(port_rets, index=rets.index)
        p_1y_cum = (1 + port_series[port_series.index >= (filled.index[-1] - pd.Timedelta(days=365.25))]).cumprod()
        history["__portfolio__"] = {"dates": [d.strftime("%Y-%m-%d") for d in p_1y_cum.index[::5]], "values": (p_1y_cum[::5] * 100).round(2).tolist()}
        cagr_p = calc_cagr((1+port_series).cumprod(), max(len(port_series)/252, 0.1)) or 0
        portfolio_risk = {"cagr_inception": cagr_p, "volatility": round(float(port_vol), 4), "sharpe_ratio": round((cagr_p-RISK_FREE_RATE)/port_vol, 2) if port_vol > 0 else 0, "var_95_daily": round(float(1.645 * (port_vol / np.sqrt(252))), 4)}
        n_sims, n_years, n_days = 2000, 10, 252; rng = np.random.default_rng(42)
        idx = rng.integers(0, len(port_rets), size=(n_sims, n_years*n_days)); sampled = port_rets[idx].reshape(n_sims, n_years, n_days)
        paths = np.column_stack([np.ones(n_sims), np.cumprod(np.prod(1+sampled, axis=2), axis=1)])
        future_value = {"median_cagr": round(float(np.percentile(paths[:,-1], 50)**(1/10)-1), 4), "p10": np.percentile(paths, 10, axis=0).tolist(), "p50": np.percentile(paths, 50, axis=0).tolist(), "p90": np.percentile(paths, 90, axis=0).tolist()}
        return {"asset_risk": asset_risk, "portfolio_risk": portfolio_risk, "future_value": future_value, "history": history, "risk_contributions": risk_contributions, "asset_vols": asset_vols, "asset_betas": asset_betas}
    except: traceback.print_exc(); return empty

def _get_advanced_intelligence(assets, portfolio, risk_level, risk_data, base_currency):
    total_val = portfolio.get("total_value", 0); beta = portfolio.get("portfolio_beta", 1.0); var_95 = portfolio.get("var_95_daily", 0); asset_vols = risk_data.get("asset_vols", {})
    kill_switch = "SYSTEM OVERLOAD: Guardrails breached. Rotate to GLD/VIG." if (var_95 > 0.035 or beta > 1.3) else None
    weighted_vol_sum = sum(assets[tk]["weight"] * asset_vols.get(tk, 0.2) for tk in assets); div_ratio = round(weighted_vol_sum / portfolio.get("volatility", 0.15), 3) if portfolio.get("volatility", 0) > 0 else 1.0
    non_base_val = sum(a["value"] for a in assets.values() if a["currency"] != base_currency); real_cagr = round(portfolio.get("cagr_inception", 0.08) - 0.025, 4)
    growth_tks = [tk for tk, a in assets.items() if a["sector"] in ("Technology", "Communication Services") or (_to_float(a.get("pe_ratio")) or 0) > 30]
    defensive_tks = [tk for tk, a in assets.items() if "Treasury" in a["name"] or "Gold" in a["name"] or tk in ("SGLN.L", "TN28.L", "GLD")]
    core_tks = [tk for tk in assets if tk not in growth_tks and tk not in defensive_tks]
    targets = {"Conservative": 0.20, "Balanced": 0.45, "Aggressive": 0.75}; target_g_w = targets.get(risk_level, 0.45)
    curr_g_w = sum(assets[tk]["weight"] for tk in growth_tks); drift = curr_g_w - target_g_w
    trade_suggestion = None; cost = 0.0
    if abs(drift) > 0.05:
        amt = abs(drift) * total_val; cost = round(amt * 0.006, 2)
        if drift > 0:
            sell_tk = max(growth_tks, key=lambda x: assets[x]["weight"]) if growth_tks else "Growth Segment"
            buy_tk = max(core_tks, key=lambda x: assets[x]["weight"]) if core_tks else "Core Segment"
            actual_trade_amt = min(amt, assets.get(sell_tk, {"value": amt})["value"])
            trade_suggestion = f"Sell {base_currency} {actual_trade_amt:,.0f} of {sell_tk} and reallocate to {buy_tk}."
        else:
            sell_source = core_tks if core_tks else [tk for tk in assets if tk not in growth_tks]
            sell_tk = max(sell_source, key=lambda x: assets[x]["weight"]) if sell_source else "Portfolio"
            buy_tk = max(growth_tks, key=lambda x: assets[x]["weight"]) if growth_tks else "Growth Proxy"
            actual_trade_amt = min(amt, assets.get(sell_tk, {"value": amt})["value"])
            trade_suggestion = f"Reduce {sell_tk} by {base_currency} {actual_trade_amt:,.0f} to fund {buy_tk} expansion."
    div_safety = {tk: max(0, min(100, round(100 - (_to_float(a.get("pe_ratio")) or 20)*1.2 - (_to_float(a.get("dividend_yield")) or 0)*400))) for tk, a in assets.items()}
    ideal_mapping = {"Conservative": {"Growth": 0.20, "Core": 0.50, "Defensive": 0.30}, "Balanced": {"Growth": 0.45, "Core": 0.40, "Defensive": 0.15}, "Aggressive": {"Growth": 0.75, "Core": 0.20, "Defensive": 0.05}}
    blueprint = ideal_mapping.get(risk_level, ideal_mapping["Balanced"])
    current_structure = {"Growth": curr_g_w, "Defensive": sum(assets[tk]["weight"] for tk in defensive_tks), "Core": 0.0}
    current_structure["Core"] = 1.0 - current_structure["Growth"] - current_structure["Defensive"]
    blueprint_actions = []; segments = {"Growth": growth_tks, "Core": core_tks, "Defensive": defensive_tks}
    for cat, target in blueprint.items():
        diff = target - current_structure[cat]
        if abs(diff) > 0.05:
            action = "Increase" if diff > 0 else "Trim"
            blueprint_actions.append({"category": cat, "action": action, "impact": f"{abs(diff)*100:.1f}% shift required", "tickers": segments[cat]})
    return {"scenarios": {"nasdaq_10": round(total_val*beta*-0.1, 2), "fx_5pct_shock": round(non_base_val*0.05, 2)}, "rebalancing": {"current": round(curr_g_w, 4), "drift": round(drift, 4), "trade": trade_suggestion, "projected_beta": round(beta - (drift*0.2), 2), "projected_sharpe": round(portfolio.get("sharpe_ratio", 0.5)*1.2, 2), "transaction_cost_estimate": cost}, "attribution": {"factor": "Growth Dominant" if curr_g_w > 0.5 else "Core/Value", "div_ratio": div_ratio, "fx_exposure_risk": "HIGH" if non_base_val/total_val > 0.7 else "LOW"}, "div_safety": div_safety, "kill_switch": kill_switch, "hedge_optimizer": {"optimal_gld_weight": 0.12, "hedge_efficiency": "HIGH" if beta > 1.1 else "LOW"}, "real_cagr": real_cagr, "outlook_score": sum((2 if a.get("sentiment", {}).get("label") == "positive" else -2 if a.get("sentiment", {}).get("label") == "negative" else 0) for a in assets.values()), "ideal_blueprint": {"target": blueprint, "current": current_structure, "actions": blueprint_actions, "segments": segments}}

def _get_market_outlook(assets, portfolio):
    points = [
        {"l": "Macro Regime", "v": "Fiscal Dominance & Rate Floor", "d": "We have transitioned into a regime of 'Fiscal Dominance' where deficit spending outpaces monetary tightening. This structural shift creates a 'Higher for Longer' floor on interest rates, significantly raising the cost of capital. In this environment, the 'Equity Risk Premium' (ERP) has compressed, making fundamental margin durability and internal cash-flow generation the only reliable factors for long-term alpha. Investors must now clear a 5%+ risk-free hurdle before any equity risk is justified.", "i": "📈"},
        {"l": "Sector Rotation", "v": "Physical AI & Infrastructure", "d": "The AI trade is rapidly evolving from the 'Training Phase' (GPUs/Compute) to the 'Inference & Physicals Phase.' Institutional capital is aggressively migrating into 'Utility Enablers'—companies specialized in data center power density, liquid cooling (e.g., Vertiv), and electrical grid modernization (e.g., Eaton). We expect a multi-year surge in demand for base-load energy sources, including Nuclear SMR infrastructure, to sustain the exponential power requirements of the global AI inference build-out.", "i": "🚀"},
        {"l": "Regional Alpha", "v": "ASEAN Corridor Alpha", "d": "Global deglobalization is no longer a trend but a structural reality. As the 'China + 1' strategy matures, permanent alpha channels are opening in 'Friend-Shoring' hubs such as the ASEAN corridors (Vietnam/India) and Mexico. Regional leaders in these zones are capturing a 'Geopolitical Risk Discount,' gaining massive market share from legacy globalists who are increasingly trapped in the crossfire of fragmented trade blocs and tariff-war escalation.", "i": "🌍"},
        {"l": "Valuation Pivot", "v": "WACC Sensitivity & FCF Yield", "d": "The era of 'Growth at Any Price' is dead. The market has pivoted to 'FCF Yield over Multiple Expansion.' With a higher Weighted Average Cost of Capital (WACC), the market is aggressively discounting 'Terminal Value' in DCF models. Institutional capital is now penalizing cash-burning growth names and prioritizing 'Quality Growth'—companies that can self-fund expansion while maintaining 20%+ FCF margins without relying on expensive capital markets.", "i": "💎"},
        {"l": "Strategic Hedge", "v": "Hard Assets & Credit Ballast", "d": "Hard Assets are now a mandatory portfolio ballast. Gold and Physical Commodities provide a structural 'Safe Haven' against currency debasement and systemic credit events. In a world of weaponized dollar-hegemony, these assets maintain low-to-negative correlations to tech-heavy portfolios, providing a vital 'Volatility Floor' and asymmetric payoff potential during non-linear tail-risk liquidation events.", "i": "🛡️"}
    ]
    comm = []; sharpe = portfolio.get("sharpe_ratio", 0)
    if sharpe > 1.2: comm.append("Exceptional risk-adjusted harvesting detected. Your allocation is efficiently capturing alpha while successfully suppressing 'Factor Crowding' noise.")
    elif sharpe < 0.7: comm.append("Suboptimal efficiency detected. Your portfolio is exhibiting 'Noisy Volatility' where multiple assets respond identically to rate shocks, leading to 'Sharpe Decay'.")
    future_outlook = "We anticipate a structural 'Broadening Trade' as mega-cap tech faces 'Multiple Compression.' Institutional capital will rotate into 'Undervalued Quality' such as secondary SaaS and industrial energy infrastructure providers. Focus on high pricing power."
    return {"points": points, "commentary": " ".join(comm), "future_outlook": future_outlook}

def _get_global_events():
    now = datetime.datetime.now()
    return [{"d": (now + pd.Timedelta(days=12)).strftime("%Y-%m-%d"), "e": "US Fed Meeting", "i": "Rate decision."}, {"d": (now + pd.Timedelta(days=18)).strftime("%Y-%m-%d"), "e": "US CPI Data", "i": "Inflation trigger."}, {"d": (now + pd.Timedelta(days=25)).strftime("%Y-%m-%d"), "e": "BoE Meeting", "i": "GBP driver."}]

def _generate_recommendations(assets, portfolio, risk_level):
    recs = []; pb = portfolio.get("portfolio_beta", 1.0); sharpe = portfolio.get("sharpe_ratio", 0.0); yld = portfolio.get("weighted_dividend_yield", 0.0)
    if pb > 1.2: recs.append({"type": "warning", "icon": "📉", "title": "High Beta Skew", "detail": f"Beta {pb:.2f} is Aggressive.", "rationale": "High sensitivity.", "action": "Rotate to VIG/GLD"})
    elif pb < 0.7: recs.append({"type": "suggestion", "icon": "📈", "title": "Beta Lag", "detail": f"Beta {pb:.2f} Defensive.", "rationale": "Bull market lag.", "action": "Expand QQQ"})
    if sharpe < 0.8: recs.append({"type": "warning", "icon": "⚖️", "title": "Efficiency Gap", "detail": f"Sharpe {sharpe:.2f} Suboptimal.", "rationale": "Noisy returns.", "action": "Add Quality"})
    if yld < 0.02: recs.append({"type": "suggestion", "icon": "💸", "title": "Income Floor", "detail": "Yield < 2%.", "rationale": "Dividends act as shock absorbers.", "action": "Add SCHD"})
    if len(assets) < 4: recs.append({"type": "warning", "icon": "🏗️", "title": "Factor Concentration", "detail": "Low breadth.", "rationale": "Idiosyncratic risk.", "action": "Add 2+ Assets"})
    return recs[:6]

async def _get_sentiment(ticker: str) -> Dict[str, Any]:
    if ticker in _sentiment_cache:
        c = _sentiment_cache[ticker]; 
        if time.time() - c["ts"] < CACHE_TTL: return c["data"]
    PROXY_MAP = {"VWRP": ["Global Stocks", "FTSE All-World"], "SPY": ["S&P 500", "Federal Reserve"], "QQQ": ["Nasdaq 100", "US Tech Sector"]}
    async def fetch_headlines(tk):
        try:
            t = yf.Ticker(tk); news = await asyncio.to_thread(lambda: t.news); 
            return [n.get("content", {}).get("title") for n in news[:10] if n.get("content", {}).get("title")]
        except: return []
    headlines = await fetch_headlines(ticker); clean_tk = ticker.split(".")[0].upper()
    if clean_tk in PROXY_MAP:
        for proxy in PROXY_MAP[clean_tk]: headlines.extend((await fetch_headlines(proxy))[:3])
    if not headlines: return {"bullish": 0.33, "bearish": 0.33, "label": "neutral", "headline_count": 0, "headlines": []}
    headlines = list(dict.fromkeys(headlines))[:20]; headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(FINBERT_API_URL, headers=headers, json={"inputs": headlines}, timeout=30.0)
            results = resp.json(); total_b = total_r = 0.0
            for res in results:
                scores = res if isinstance(res, list) else [res]
                for s in scores:
                    if s.get("label") == "positive": total_b += s.get("score", 0)
                    elif s.get("label") == "negative": total_r += s.get("score", 0)
            avg_b, avg_r = round(total_b/max(len(headlines),1), 3), round(total_r/max(len(headlines),1), 3)
            data = {"bullish": avg_b, "bearish": avg_r, "label": "positive" if avg_b > avg_r and avg_b > 0.35 else "negative" if avg_r > avg_b and avg_r > 0.35 else "neutral", "headline_count": len(headlines), "headlines": headlines, "sentiment_delta": round(avg_b - 0.45, 2), "exhaustion_alert": avg_b < 0.25}
            _sentiment_cache[ticker] = {"ts": time.time(), "data": data}; return data
    except: return {"bullish": 0.33, "bearish": 0.33, "label": "neutral", "headline_count": 0, "headlines": []}

@app.get("/api/config")
async def get_config():
    return {"apiKey": os.getenv("FB_API_KEY"), "authDomain": os.getenv("FB_AUTH_DOMAIN"), "projectId": os.getenv("FB_PROJECT_ID"), "storageBucket": os.getenv("FB_STORAGE_BUCKET"), "messagingSenderId": os.getenv("FB_MESSAGING_SENDER_ID"), "appId": os.getenv("FB_APP_ID"), "measurementId": os.getenv("FB_MEASUREMENT_ID")}

@app.post("/api/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    try:
        holdings = request.holdings; tickers = list(holdings.keys()); base_curr = request.base_currency or "GBP"
        async def _fetch_info(ticker):
            try:
                t = yf.Ticker(ticker); info = await asyncio.to_thread(lambda: t.info) or {}
                p = _to_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0)
                return ticker, info, p
            except: return ticker, {}, 0
        res_info = await asyncio.gather(*[_fetch_info(t) for t in tickers])
        infos = {tk: info for tk, info, p in res_info}; usd_gbp = await _get_fx_rate("USD", base_curr); gbp_usd = 1.0 / usd_gbp if usd_gbp else 1.0
        fallbacks = request.fallback_prices or {}; assets = {}
        for tk, info, p in res_info:
            default_curr = "GBP" if tk.endswith(".L") else "USD"; curr = safe_fetch(info, "currency", default_curr)
            if curr in ("GBp", "GBX"): p /= 100.0; curr = "GBP"
            if (not p or p <= 0) and tk in fallbacks:
                p = fallbacks[tk]; 
                if tk.endswith(".L") and p > 500: p /= 100.0
            fx_to_base = usd_gbp if (curr == "USD" and base_curr == "GBP") else gbp_usd if (curr == "GBP" and base_curr == "USD") else 1.0
            val_in_base = round(p * holdings[tk] * fx_to_base, 2)
            assets[tk] = {"name": safe_fetch(info, "longName", tk), "price": round(p, 2), "quantity": holdings[tk], "value": val_in_base, "currency": curr, "dividend_yield": _round(info.get("dividendYield")), "pe_ratio": _round(info.get("trailingPE")), "beta": _round(info.get("beta")), "sector": safe_fetch(info, "sector", "N/A"), "institutional_flow_score": round(_to_float(info.get("heldPercentInstitutions", 0.5))*100, 1)}
        total_val = sum(a["value"] for a in assets.values()); total_val = total_val if total_val > 0 else 1.0
        for a in assets.values(): a["weight"] = round(a["value"]/total_val, 4)
        full_prices = await _fetch_full_history(list(set(tickers + ["SPY"]))); risk_data = _compute_risk_history_from_prices(full_prices, tickers, {tk: assets[tk]["weight"] for tk in tickers}); sentiments = await asyncio.gather(*[_get_sentiment(t) for t in tickers])
        for tk, s in zip(tickers, sentiments): 
            assets[tk]["sentiment"] = s; assets[tk]["risk_contribution_pct"] = round(risk_data.get("risk_contributions", {}).get(tk, 0) * 100, 2)
        p_summary = {**risk_data["portfolio_risk"], "total_value": round(total_val, 2), "weighted_dividend_yield": round(sum((_to_float(a["dividend_yield"]) or 0)*a["weight"] for a in assets.values()), 4), "portfolio_beta": round(sum((_to_float(a["beta"]) or 1)*a["weight"] for a in assets.values()), 2), "asset_count": len(tickers), "currency": base_curr}
        advanced = _get_advanced_intelligence(assets, p_summary, request.risk_level, risk_data, base_curr)
        return {"portfolio": p_summary, "assets": assets, "risk": risk_data, "advanced_intel": advanced, "recommendations": _generate_recommendations(assets, p_summary, request.risk_level), "events": {tk: {"earnings_date": None, "ex_dividend_date": None} for tk in tickers}, "global_macro_events": _get_global_events(), "market_outlook": _get_market_outlook(assets, p_summary)}
    except Exception as e: traceback.print_exc(); raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health(): return {"status": "ok"}

@https_fn.on_request(memory=options.MemoryOption.GB_2, timeout_sec=300, region="us-central1")
def vesper_api(req: https_fn.Request) -> https_fn.Response:
    from fastapi.testclient import TestClient
    client = TestClient(app); path = req.path
    if path.startswith("/vesper_api"): path = path[len("/vesper_api"):]
    if not path: path = "/"
    response = client.request(method=req.method, url=path, params=req.args, headers=dict(req.headers), content=req.data)
    return https_fn.Response(response.content, status=response.status_code, headers=dict(response.headers))
