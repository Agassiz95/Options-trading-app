'''
This script solves the Black-Scholes equation for call and put options.
After running the script, enter the stock ticker, strike price, expiration date, and option type.
It returns the current stock price, implied volatility, time to maturity in years,
European Option Price, and American Option Price.

The result is a fair theoretical value for comparison with market prices.

MIT Open Source License:

Copyright (c) 2025 Robert Chance

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

# --------------------------------------
# Black-Scholes for European Call or Put
# --------------------------------------
def european_option_bs(S, K, T, r, sigma, option_type='call'):
    S = np.clip(S, 1e-8, None)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --------------------------------------
# Thomas algorithm for tridiagonal system
# --------------------------------------
def thomas_algorithm(A, d):
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n - 1)
    for i in range(n):
        b[i] = A[i, i]
        if i > 0:
            a[i] = A[i, i - 1]
        if i < n - 1:
            c[i] = A[i, i + 1]

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x

# --------------------------------------
# American option via Crank-Nicolson FD
# --------------------------------------
def american_option_fd(S_max, K, T, r, sigma, S_steps=200, T_steps=1000, option_type='call'):
    dS = S_max / S_steps
    dt = T / T_steps
    S = np.linspace(0, S_max, S_steps + 1)
    V = np.maximum(0, (S - K) if option_type == 'call' else (K - S))
    payoff = V.copy()

    alpha = 0.25 * dt * (sigma ** 2 * (np.arange(S_steps + 1)) ** 2 - r * np.arange(S_steps + 1))
    beta = -0.5 * dt * (sigma ** 2 * (np.arange(S_steps + 1)) ** 2 + r)
    gamma = 0.25 * dt * (sigma ** 2 * (np.arange(S_steps + 1)) ** 2 + r * np.arange(S_steps + 1))

    M1 = np.zeros((S_steps - 1, S_steps - 1))
    M2 = np.zeros((S_steps - 1, S_steps - 1))

    for i in range(S_steps - 1):
        if i > 0:
            M1[i, i - 1] = -alpha[i + 1]
            M2[i, i - 1] = alpha[i + 1]
        M1[i, i] = 1 - beta[i + 1]
        M2[i, i] = 1 + beta[i + 1]
        if i < S_steps - 2:
            M1[i, i + 1] = -gamma[i + 1]
            M2[i, i + 1] = gamma[i + 1]

    for t in range(T_steps):
        rhs = M2 @ V[1:-1]
        rhs[0] += alpha[1] * V[0] + alpha[1] * V[0]
        rhs[-1] += gamma[-2] * V[-1] + gamma[-2] * V[-1]
        V_inner = thomas_algorithm(M1, rhs)
        V[1:-1] = np.maximum(V_inner, payoff[1:-1])
        V[0] = 0
        if option_type == 'call':
            V[-1] = S_max - K * np.exp(-r * (T - t * dt))
        else:
            V[-1] = 0

    return S, V

# --------------------------------------
# Fetch stock data and volatility
# --------------------------------------
def fetch_data(ticker, expiration):
    stock = yf.Ticker(ticker)
    S0 = stock.history(period="1d")['Close'].iloc[-1]
    hist = stock.history(period="60d")['Close']
    returns = np.log(hist / hist.shift(1)).dropna()
    sigma = returns.std() * np.sqrt(252)
    T = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.today()).days / 365
    T = max(T, 1 / 365)
    return S0, sigma, T

def get_risk_free_rate(default=0.045):
    try:
        treasury = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1]
        return treasury / 100
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch risk-free rate. Using default {default:.2%}.")
        return default

def fetch_market_price(ticker, strike, expiration, option_type):
    stock = yf.Ticker(ticker)
    try:
        exp_chain = stock.option_chain(expiration)
        chain = exp_chain.calls if option_type == 'call' else exp_chain.puts
        row = chain[chain['strike'] == strike]
        if not row.empty:
            return (
                float(row.iloc[0]['lastPrice']),
                float(row.iloc[0]['bid']),
                float(row.iloc[0]['ask']),
            )
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch market option price: {e}")
    return None, None, None

def european_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100
    return delta, gamma, theta, vega, rho


def make_trade_suggestion(S, K, T, sigma, market_price, model_price, option_type):
    if market_price is None:
        return "Market price unavailable.", None

    # Mispricing and z-score
    mispricing = market_price - model_price

    # Estimate ITM probability from Black-Scholes d2
    d2 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)) - sigma * np.sqrt(T)
    pop = norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)

    # Suggest action based on mispricing magnitude
    if mispricing > 0.5:
        suggestion = f"️Overpriced by ${mispricing:.2f} — consider selling"
    elif mispricing < -0.5:
        suggestion = f"Underpriced by ${-mispricing:.2f} — consider buying"
    else:
        suggestion = "ℹFairly priced — no strong edge detected"

    return suggestion, pop

# --------------------------------------
# Main script prompt
# --------------------------------------
if __name__ == "__main__":
    print("Option Pricing Tool (European and American Options)\n")

    print(f" Note: The following information is for informational purposes only.\n Use this options tool at "
          f"your own risk! The maker of this tool is not \n liable for any lost capital during any trading."
          f" However, the maker \n of this tool does appreciate donations! Venmo: @Rob_C_95\n")

    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    strike = float(input("Enter strike price: "))
    expiration = input("Enter expiration date (YYYY-MM-DD): ").strip()
    option_type = input("Enter option type (call/put): ").strip().lower()
    assert option_type in ["call", "put"], "Option type must be 'call' or 'put'"
    r = get_risk_free_rate()

    S0, sigma, T = fetch_data(ticker, expiration)

    euro_price = european_option_bs(S0, strike, T, r, sigma, option_type)
    S_am, V_am = american_option_fd(S_max=2 * S0, K=strike, T=T, r=r, sigma=sigma, option_type=option_type)
    amer_price = np.interp(S0, S_am, V_am)

    # Fetch market prices
    market_last, market_bid, market_ask = fetch_market_price(ticker, strike, expiration, option_type)

    # Compute Greeks
    delta, gamma, theta, vega, rho = european_greeks(S0, strike, T, r, sigma, option_type)

    print(f"\nCurrent stock price (S0): ${S0:.2f}")
    print(f"Implied volatility (σ): {sigma:.2%}")
    print(f"Time to maturity (T): {T:.3f} years")
    print(f"European {option_type.capitalize()} Price (Black-Scholes): ${euro_price:.2f}")
    print(f"American {option_type.capitalize()} Price (Crank-Nicolson FD): ${amer_price:.2f}")

    if market_last is not None:
        print(f"Market Last / Bid / Ask: ${market_last:.2f} / ${market_bid:.2f} / ${market_ask:.2f}")
    else:
        print("Market option data unavailable.")

    print("\nOption Greeks (European model):")
    print(f"  Delta: {delta:.4f}")
    print(f"  Gamma: {gamma:.4f}")
    print(f"  Theta (per day): {theta:.4f}")
    print(f"  Vega (per 1% IV): {vega:.4f}")
    print(f"  Rho (per 1% rate): {rho:.4f}")

    suggestion, pop = make_trade_suggestion(S0, strike, T, sigma, market_last, euro_price, option_type)
    print(f"\n Trade Suggestion: {suggestion}")
    if pop is not None:
        print(f" Estimated Probability of Profit (ITM): {pop:.2%}")
        print(
            "⚠️ Note: This is the probability of expiring in-the-money — not the probability of net profit (which depends on premium paid).")

    # Plot both options
    S_range = np.linspace(0.5 * S0, 1.5 * S0, 200)
    euro_vals = european_option_bs(S_range, strike, T, r, sigma, option_type)

    plt.figure(figsize=(10, 6))
    plt.plot(S_range, euro_vals, label=f"European {option_type.capitalize()} (BS)", linewidth=2)
    plt.plot(S_am, V_am, label=f"American {option_type.capitalize()} (FD)", linewidth=2)
    plt.axvline(S0, color='green', linestyle='--', label=f"S0 = ${S0:.2f}")
    plt.axvline(strike, color='red', linestyle='--', label=f"Strike = ${strike:.2f}")
    plt.title(f"{ticker.upper()} {option_type.capitalize()} Option Prices")
    plt.xlabel("Stock Price")
    plt.ylabel("Option Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
