# Vesper — Portfolio Intelligence Tool

A high-performance, real-time portfolio analysis system built with **FastAPI**, **FinBERT AI**, and **Vanilla JS/Tailwind**.

## Features

- **Weighted Portfolio Metrics**: Total value, dividend yield, expense ratio, and beta
- **Correlation Matrix**: 1-year daily return Pearson correlations for diversification analysis
- **FinBERT Sentiment Analysis**: Real-time AI-driven sentiment on news headlines with 60-min cache
- **Interactive Dashboard**: Doughnut allocation chart, sentiment gauges, correlation heatmap
- **Resilient Data Fetching**: SafeFetch pattern gracefully handles missing data
- **Async Processing**: High-concurrency backend with thread pool for blocking operations

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI 0.115.5 + Uvicorn |
| **Data** | yfinance 0.2.50, pandas 2.2.3, numpy 1.26.4 |
| **AI/ML** | transformers 4.47.0, ProsusAI/FinBERT |
| **Frontend** | HTML5, Tailwind CSS (CDN), Chart.js |
| **Validation** | Pydantic 2.10.3 |
| **Python** | 3.11+ |

## Quick Start

### 1. Clone / Enter the Directory
```bash
cd /Users/sanjaysharma/vesper
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note on PyTorch**: The `requirements.txt` includes CPU-only torch (~250 MB). For GPU support, replace with:
```bash
pip install torch==2.5.1  # will pull the default CUDA-enabled wheel
```

### 4. Run the Server
```bash
python main.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 [Press ENTER to quit]
🚀  Vesper starting — loading ProsusAI/finbert …
✅  FinBERT loaded and ready.
```

**First run**: FinBERT model (~500 MB) downloads automatically. This may take 30–60 seconds.

### 5. Open the Dashboard
Navigate to **http://localhost:8000** in your browser.

---

## Usage

### Portfolio Builder
1. Enter ticker symbols (e.g., `AAPL`, `MSFT`, `SPY`, `QQQ`)
2. Specify quantities (supports fractional shares)
3. Click **Analyse Portfolio**

### Results
- **Summary Cards**: Total value, portfolio beta, weighted yield, expense ratio
- **Allocation Chart**: Doughnut visualization of holdings by value
- **Sentiment Panel**: Per-asset bullish/bearish/neutral scores from FinBERT
- **Correlation Heatmap**: Color-coded matrix showing asset diversification
- **Asset Details**: 10+ metrics per holding (P/E, beta, 52W range, sector, etc.)

---

## API Endpoints

### `POST /api/analyze`
Analyse a portfolio of holdings.

**Request:**
```json
{
  "holdings": {
    "AAPL": 10,
    "MSFT": 5,
    "SPY": 3
  }
}
```

**Response:**
```json
{
  "portfolio": {
    "total_value": 5234.67,
    "weighted_dividend_yield": 0.0145,
    "weighted_expense_ratio": 0.0015,
    "portfolio_beta": 1.05,
    "asset_count": 3
  },
  "assets": {
    "AAPL": {
      "price": 245.67,
      "value": 2456.70,
      "weight": 0.469,
      "sentiment": { "bullish": 0.65, "bearish": 0.15, "neutral": 0.20, "label": "positive" },
      ...
    }
  },
  "correlation": {
    "tickers": ["AAPL", "MSFT", "SPY"],
    "matrix": [[1.0, 0.87, 0.92], [0.87, 1.0, 0.89], [0.92, 0.89, 1.0]]
  }
}
```

### `GET /health`
Quick liveness probe.

**Response:**
```json
{
  "status": "ok",
  "finbert_ready": true
}
```

---

## Code Structure

```
vesper/
├── main.py           # FastAPI backend (484 lines)
│   ├── Lifespan → FinBERT model loading
│   ├── PortfolioRequest → Pydantic validation
│   ├── Correlation Engine → 1-year Pearson matrix
│   ├── Beta Engine → per-asset vs SPY
│   ├── Sentiment Engine → FinBERT + 60-min cache
│   └── POST /api/analyze → Main endpoint
│
├── index.html        # Single-file SPA (675 lines)
│   ├── Header + Health indicator
│   ├── Portfolio builder form
│   ├── Summary cards
│   ├── Allocation chart (Chart.js)
│   ├── Sentiment panel
│   ├── Correlation heatmap
│   └── Asset detail cards
│
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Caching & Performance

**Sentiment Cache**: FinBERT results are cached in-memory for 60 minutes per ticker. Subsequent requests for the same ticker reuse cached sentiment scores, avoiding redundant ~5–10s inference runs.

**Thread Pool**: Blocking operations (yfinance fetches, FinBERT inference) run in a `ThreadPoolExecutor(max_workers=12)` to keep the async event loop responsive.

---

## Error Handling

- **Invalid tickers**: Returns `"N/A"` for missing/unfetchable fields
- **Missing prices**: Raises HTTP 422 with clear error message
- **Insufficient correlation history**: Returns error in `correlation.error`
- **Request validation**: Pydantic validates holdings (1–20 tickers, positive quantities)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: transformers` | Run `pip install -r requirements.txt` |
| FinBERT still loading after 2 min | Model is downloading (~500 MB). Check internet; model cache often stored in `~/.cache/huggingface` |
| Port 8000 already in use | Change port in `main.py`: `uvicorn.run(..., port=8001)` |
| Sentiment all "neutral" | yfinance news may be unavailable for that ticker; check if ticker is valid |
| No price found error | Verify ticker symbols are correct (e.g., `SPY` not `SPA`) and markets are open |

---

## Future Enhancements

- **Real-time updates**: WebSocket connection for live price/sentiment streaming
- **Watchlist persistence**: Save/load portfolios to local storage or backend DB
- **Advanced analytics**: Risk decomposition, factor exposures, backtesting
- **Email alerts**: Custom notifications on correlation spikes or sentiment shifts
- **Mobile responsive**: Optimize for tablets and phones

---

## License

Educational use only. Not financial advice. Always consult a qualified financial advisor.

---

## Author

Senior Full-Stack Fintech Engineer & Python Data Architect

---

## References

- **yfinance**: https://github.com/ranaroussi/yfinance
- **FinBERT**: https://github.com/ProsusAI/finBERT
- **FastAPI**: https://fastapi.tiangolo.com/
- **Chart.js**: https://www.chartjs.org/
- **Tailwind CSS**: https://tailwindcss.com/
