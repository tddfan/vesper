import yfinance as yf
import json

ticker = "AAPL"
t = yf.Ticker(ticker)
news = t.news
print(f"News count for {ticker}: {len(news)}")
if news:
    print("First news item keys:", news[0].keys())
    print("First news item title:", news[0].get("title"))

# Mock headlines for testing inference format if needed
headlines = [n.get("title") for n in news[:5] if n.get("title")]
print("Headlines:", json.dumps(headlines, indent=2))
