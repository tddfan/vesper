# Vesper Firebase Deployment Guide

This document outlines the structural changes required to migrate Vesper from a local FastAPI server to **Firebase Cloud Functions** (backend) and **Firebase Hosting** (frontend).

---

## Current Architecture

```
Local Machine
├── localhost:8000 (Uvicorn)
├── main.py (FastAPI)
└── index.html (Served alongside)
```

**Key components:**
- FastAPI with lifespan context for model loading
- In-memory sentiment cache (60 min TTL)
- Thread pool for blocking I/O
- Synchronous global state: `_finbert`, `_sentiment_cache`, `_executor`

---

## Target Architecture

```
Google Cloud
├── Firebase Cloud Functions
│   ├── Function 1: POST /api/analyze
│   ├── Function 2: GET /health
│   └── Firestore (for persistent cache)
│
└── Firebase Hosting
    └── index.html (SPA with hardcoded API URL)
```

---

## Phase 1: Backend Migration → Cloud Functions

### 1.1 Refactor for Stateless Execution

**Issue**: Cloud Functions scale horizontally; in-memory caches don't persist across function instances.

**Solution**:
- Replace `_sentiment_cache` (dict) with **Firestore** collection
- Replace `_finbert` (global) with **lazy loading** per request or **warm-up** triggers
- Remove `_executor` thread pool (Cloud Functions handle concurrency differently)

### 1.2 Create Cloud Function (Python 3.11)

**New file: `functions/main.py`**

```python
import functions_framework
import asyncio
import json
from google.cloud import firestore
from transformers import pipeline
import yfinance as yf
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Firestore client
db = firestore.Client()

# Global (loaded once per warm instance)
_finbert = None

def load_finbert():
    global _finbert
    if _finbert is None:
        _finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
    return _finbert

@functions_framework.http
def analyze_portfolio(request):
    """Cloud Function: POST /api/analyze"""
    
    # CORS headers
    if request.method == 'OPTIONS':
        return ('', 204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        })

    try:
        data = request.get_json()
        holdings = data.get("holdings", {})
        
        # Validate holdings (same as local version)
        if not holdings or len(holdings) > 20:
            return json.dumps({"detail": "Invalid holdings"}), 422
        
        # Run async function using asyncio.run()
        result = asyncio.run(_analyze_async(holdings))
        
        return json.dumps(result), 200, {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json',
        }
    
    except Exception as e:
        return json.dumps({"detail": str(e)}), 500, {
            'Access-Control-Allow-Origin': '*',
        }

async def _analyze_async(holdings):
    """Main analysis logic (refactored from local main.py)"""
    
    tickers = list(holdings.keys())
    
    # 1. Fetch ticker info
    infos = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            infos[ticker] = t.info or {}
        except:
            infos[ticker] = {}
    
    # 2. Build asset data (same as local)
    assets = {}
    for ticker in tickers:
        info = infos.get(ticker, {})
        price_raw = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice")
        price = float(price_raw) if price_raw else 0.0
        qty = holdings[ticker]
        value = round(price * qty, 2)
        
        assets[ticker] = {
            "price": round(price, 2),
            "value": value,
            "quantity": qty,
            "weight": 0.0,
            "dividend_yield": info.get("dividendYield", "N/A"),
            "expense_ratio": info.get("expenseRatio") or info.get("annualReportExpenseRatio", "N/A"),
            # ... (all other fields from local version)
        }
    
    # 3. Calculate portfolio totals
    total_value = sum(a["value"] for a in assets.values() if isinstance(a["value"], (int, float)))
    if total_value == 0:
        raise ValueError("Could not determine price for any ticker.")
    
    for ticker in assets:
        assets[ticker]["weight"] = round(assets[ticker]["value"] / total_value, 6)
    
    # 4. Correlation matrix (blocking, but OK in Cloud Functions context)
    correlation = _compute_correlation(tickers)
    
    # 5. Sentiment (with Firestore cache)
    sentiments = {}
    for ticker in tickers:
        sentiments[ticker] = await _get_sentiment_with_firestore_cache(ticker)
        assets[ticker]["sentiment"] = sentiments[ticker]
    
    return {
        "portfolio": {
            "total_value": total_value,
            "weighted_dividend_yield": _weighted_avg(assets, "dividend_yield"),
            "weighted_expense_ratio": _weighted_avg(assets, "expense_ratio"),
            "portfolio_beta": _weighted_avg(assets, "beta"),
            "asset_count": len(tickers),
        },
        "assets": assets,
        "correlation": correlation,
    }

async def _get_sentiment_with_firestore_cache(ticker: str):
    """Fetch sentiment; check Firestore cache first (60 min TTL)"""
    
    # Check Firestore
    cache_doc = db.collection('sentiment_cache').document(ticker).get()
    if cache_doc.exists:
        data = cache_doc.to_dict()
        if time.monotonic() - data['ts'] < 3600:  # 60 min TTL
            return data['result']
    
    # Cache miss → run FinBERT
    finbert = load_finbert()
    # ... (same headline fetching + FinBERT logic)
    result = { "bullish": ..., "bearish": ..., "neutral": ..., "label": ... }
    
    # Store in Firestore
    db.collection('sentiment_cache').document(ticker).set({
        'result': result,
        'ts': time.monotonic(),
    })
    
    return result

def _compute_correlation(tickers):
    """Unchanged from local version"""
    # ...
    pass

@functions_framework.http
def health(request):
    """Cloud Function: GET /health"""
    if request.method == 'OPTIONS':
        return ('', 204)
    
    try:
        finbert = load_finbert()
        return json.dumps({
            "status": "ok",
            "finbert_ready": finbert is not None,
        }), 200, {
            'Access-Control-Allow-Origin': '*',
        }
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}), 500
```

### 1.3 Setup Cloud Functions Deployment

**File: `functions/requirements.txt`**

```
fastapi==0.115.5
pydantic==2.10.3
yfinance==0.2.50
pandas==2.2.3
numpy==1.26.4
transformers==4.47.0
torch==2.5.1
google-cloud-firestore==2.14.0
functions-framework==3.5.0
```

**Deploy function:**
```bash
cd functions
gcloud functions deploy analyze-portfolio \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point analyze_portfolio \
  --region us-central1 \
  --memory 2GB \
  --timeout 300
```

---

## Phase 2: Frontend Migration → Firebase Hosting

### 2.1 Update API Endpoint

**In `index.html`, update the API URL:**

```javascript
// Before (local):
const API_URL = '/api/analyze'

// After (Firebase):
const API_URL = 'https://us-central1-[PROJECT-ID].cloudfunctions.net/analyze-portfolio'
```

Or use environment variables:
```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/analyze'
```

### 2.2 Remove Server Routing

Remove the `GET /` endpoint from main.py since Firebase Hosting serves `index.html` directly.

### 2.3 Deploy to Firebase Hosting

**File: `firebase.json`**

```json
{
  "hosting": {
    "public": "public",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [
      {
        "source": "**",
        "destination": "/index.html"
      }
    ]
  }
}
```

**Setup:**
```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login
firebase login

# Initialize
firebase init hosting

# Copy index.html to public/
mkdir public
cp index.html public/

# Deploy
firebase deploy
```

---

## Phase 3: Configuration & Security

### 3.1 Firestore Security Rules

**firestore.rules:**

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Read/write to sentiment_cache collection only from Cloud Function
    match /sentiment_cache/{document=**} {
      allow read, write: if false; // Closed by default
    }
  }
}
```

Firestore accessed only by Cloud Function service account.

### 3.2 Cloud Function Permissions

```bash
# Grant Cloud Function access to Firestore
gcloud projects add-iam-policy-binding [PROJECT-ID] \
  --member=serviceAccount:[PROJECT-ID]@appspot.gserviceaccount.com \
  --role=roles/datastore.user
```

### 3.3 CORS Configuration

The Cloud Function must return CORS headers. Already included in above code:

```python
'Access-Control-Allow-Origin': '*',
'Access-Control-Allow-Methods': 'POST, OPTIONS',
'Access-Control-Allow-Headers': 'Content-Type',
```

---

## Phase 4: Testing & Monitoring

### 4.1 Local Development

```bash
# Install Firebase Emulator Suite
npm install -g firebase-tools
firebase emulators:start

# In another terminal, test Cloud Function locally
```

### 4.2 Monitoring

- **Cloud Logging**: View function logs in *Cloud Console > Logs*
- **Cloud Trace**: Monitor latency across services
- **Error Reporting**: Automatic error tracking

### 4.3 Benchmarks

| Metric | Local | Firebase |
|--------|-------|----------|
| Cold start | — | ~15–30s (first run) |
| Warm start | <1s | ~1–2s |
| Sentiment cache | In-memory (60 min) | Firestore (TTL or manual cleanup) |
| Cost | $0 (local) | ~$0.40/M invocations + Firestore storage |

---

## Migration Checklist

- [ ] Refactor `main.py` → `functions/main.py` (stateless)
- [ ] Replace in-memory cache → Firestore
- [ ] Create `functions/requirements.txt`
- [ ] Deploy Cloud Functions
- [ ] Update `index.html` API_URL
- [ ] Create `firebase.json` config
- [ ] Deploy to Firebase Hosting
- [ ] Set Firestore security rules
- [ ] Grant Cloud Function Firestore permissions
- [ ] Test `/api/analyze` endpoint from frontend
- [ ] Test `/health` endpoint
- [ ] Monitor logs and errors
- [ ] Set up billing alerts (optional)

---

## Estimated Timeline

| Phase | Time |
|-------|------|
| Backend refactor | 2–3 hours |
| Cloud Functions deploy | 30 min |
| Frontend updates | 15 min |
| Firebase Hosting setup | 30 min |
| Testing & debugging | 1–2 hours |
| **Total** | **5–7 hours** |

---

## Phase 5: Cost Optimization (Zero-Cost Maintenance)

To ensure you stay within the **0.5 GB free storage limit** and never pay for old build images:

### 5.1 Automated Purge (Google Cloud Console)
1. Go to [Artifact Registry](https://console.cloud.google.com/artifacts/repository/vesper-1e6b7/us-central1/gcf-artifacts).
2. Click on the **`gcf-artifacts`** repository.
3. Click **"Cleanup Policies"** (top menu).
4. Click **"Add Value"** and create this rule:
   - **Name**: `purge-old-versions`
   - **Action**: `Delete`
   - **Condition**: `Keep most recent versions`
   - **Value**: `1`
5. Click **Save**.

### 5.2 Deployment Practice
When you run `firebase deploy`, you will see a prompt:
`? How many days do you want to keep container images before they're deleted?`
**Always type `1` and press Enter.** 

This ensures Google purges the 80MB image from the previous build immediately, keeping your total usage at ~80MB (well below the 500MB free limit).

---

## References

- **Cloud Functions Python 3.11**: https://cloud.google.com/functions/docs/runtime/python/overview
- **Firestore Documentation**: https://cloud.google.com/firestore/docs
- **Firebase Hosting**: https://firebase.google.com/docs/hosting
- **gcloud CLI**: https://cloud.google.com/sdk/gcloud
- **Cloud Functions Limits**: Max memory 16GB, max timeout 3600s (very generous for Vesper)
