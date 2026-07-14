import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as ps
ticker = 'AAPL'
df = yf.download(ticker, start="2023-01-01", end="2024-01-01",
                 multi_level_index=False)
df['Log_Return'] = np.log(df['Close'] /
                          df['Close'].shift(1))

print(df['Log_Return'].head())
print(df.head())
mu = df['Log_Return'].mean()
sigma = df['Log_Return'].std()
print(f"daglig drift (medel): {mu})")
print(f"daglig (volatilitet): {sigma}")
last_price = df['Close'].iloc[-1]
t_intervals = 30
iterations = 1000
daily_simple_returns = np.exp(np.random.normal(
    mu, sigma, (t_intervals, iterations)))
price_list = np.zeros_like(daily_simple_returns)
price_list[0] = last_price
for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_simple_returns[t]

plt.figure(figsize=(10, 6))
plt.plot(price_list)
plt.title(f"monte carlo simulering för {ticker} (30 dagar)")
plt.xlabel("dagar")
plt.ylabel("pris USD")
plt.savefig("monte_carlo.png")
var_5 = np.percentile(price_list[-1], 5)
print(f"Value at Risk(5% konfidens):{var_5:.2f} USD")
plt.show()
