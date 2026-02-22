import yfinance as yf
import json

ticker = "AAPL"
t = yf.Ticker(ticker)
news = t.news
if news:
    print("Content keys:", news[0]['content'].keys())
    print("Title:", news[0]['content'].get('title'))
    print("Summary:", news[0]['content'].get('summary'))
