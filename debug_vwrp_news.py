import yfinance as yf
ticker = "VWRP.L"
t = yf.Ticker(ticker)
news = t.news
print(f"News count for {ticker}: {len(news)}")
if news:
    print("First item content keys:", news[0]['content'].keys())
    print("Title:", news[0]['content'].get('title'))
