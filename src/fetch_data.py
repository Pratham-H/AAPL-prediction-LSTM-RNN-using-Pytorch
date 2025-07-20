import yfinance as yf

ticker = yf.Ticker("AAPL")

data = ticker.history(start="2000-01-01", end="2023-01-01")

data.to_csv("AAPL_adjusted.csv")
