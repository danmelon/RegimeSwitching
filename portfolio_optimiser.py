import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize

# ── Asset universe ────────────────────────────────────────────────
ASSETS = ["spy", "qqq", "tlt", "gld"]

def get_asset_returns(prices):
    """Compute daily log returns for each asset in the universe."""
    asset_prices = prices[ASSETS]
    returns = np.log(asset_prices / asset_prices.shift(1)).dropna()
    return returns

def compute_regime_statistics(asset_returns, regimes):
    """
    Split returns into bull and bear days using HMM regime labels.
    Compute mean return vector and covariance matrix for each regime.
    """
    aligned = asset_returns.reindex(regimes.index).dropna()
    regimes_aligned = regimes.reindex(aligned.index)

    bull_returns = aligned[regimes_aligned == 0]
    bear_returns = aligned[regimes_aligned == 1]

    stats = {}
    for name, subset in [("bull", bull_returns), ("bear", bear_returns)]:
        mu    = subset.mean() * 252              # annualised mean returns
        sigma = subset.cov()  * 252              # annualised covariance matrix
        stats[name] = {"mu": mu, "sigma": sigma, "n_days": len(subset)}
        print(f"\n{name.capitalize()} regime — {len(subset)} days")
        print(f"  Annualised returns:")
        for asset in ASSETS:
            print(f"    {asset.upper():4s}: {mu[asset]*100:.1f}%")
        print(f"  Correlation matrix:")
        print(subset.corr().round(2))

    return stats

def max_sharpe_portfolio(mu, sigma, risk_free=0.02):
    """
    Find the portfolio weights that maximise the Sharpe ratio.
    Uses scipy minimise on the negative Sharpe (minimising = maximising).
    Constraints: weights sum to 1, all weights between 0 and 1 (no shorting).
    """
    n = len(mu)

    def neg_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol    = np.sqrt(np.dot(w, np.dot(sigma, w)))
        return -(port_return - risk_free) / port_vol

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * n
    w0          = np.ones(n) / n  # equal weight starting point

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    weights = pd.Series(result.x, index=mu.index)
    return weights

def run(train, test, prices, hmm_regimes):

    # ── 1. Asset returns ──────────────────────────────────────────
    asset_returns = get_asset_returns(prices)

    # ── 2. Regime statistics on training set ──────────────────────
    print("=" * 55)
    print("REGIME-CONDITIONAL STATISTICS (train set)")
    print("=" * 55)
    train_regimes = hmm_regimes.reindex(train.index).dropna()
    stats = compute_regime_statistics(asset_returns, train_regimes)

    # ── 3. Optimise portfolios for each regime ────────────────────
    print("\n" + "=" * 55)
    print("OPTIMAL PORTFOLIO WEIGHTS")
    print("=" * 55)

    w_bull = max_sharpe_portfolio(stats["bull"]["mu"], stats["bull"]["sigma"])
    w_bear = max_sharpe_portfolio(stats["bear"]["mu"], stats["bear"]["sigma"])

    print("\nBull portfolio (max Sharpe):")
    for asset, w in w_bull.items():
        print(f"  {asset.upper():4s}: {w*100:.1f}%")

    print("\nBear portfolio (max Sharpe):")
    for asset, w in w_bear.items():
        print(f"  {asset.upper():4s}: {w*100:.1f}%")

    # ── 4. Blended backtest on test set ───────────────────────────
    print("\n" + "=" * 55)
    print("BLENDED PORTFOLIO BACKTEST (test set)")
    print("=" * 55)

    test_returns  = asset_returns.reindex(test.index).dropna()
    test_regimes  = hmm_regimes.reindex(test_returns.index).dropna()
    test_returns  = test_returns.reindex(test_regimes.index)

    # Bull probability from HMM — 0 = bull, 1 = bear
    # P(bull) = 1 when regime == 0, 0 when regime == 1
    p_bull = (1 - test_regimes).shift(1).fillna(1)  # shift to avoid lookahead
    p_bear = 1 - p_bull

    # Blended weights each day
    # portfolio return = sum of (blended weight × asset return) across assets
    port_returns = pd.Series(index=test_returns.index, dtype=float)
    for date in test_returns.index:
        pb = p_bull.loc[date]
        pb_bear = p_bear.loc[date]
        w_blend = pb * w_bull.values + pb_bear * w_bear.values
        port_returns.loc[date] = np.dot(w_blend, test_returns.loc[date].values)

    # Benchmarks
    spy_returns  = test_returns["spy"]
    equal_weight = test_returns[ASSETS].mean(axis=1)

    def sharpe(r):
        return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0

    def max_drawdown(r):
        cum = (1 + r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        return dd.min()

    def total_return(r):
        return (1 + r).prod() - 1

    print(f"\n{'Strategy':<28} {'Return':>8} {'Sharpe':>8} {'Max DD':>8}")
    print("-" * 55)
    for label, r in [
        ("Blended regime portfolio", port_returns),
        ("Equal weight (SPY/QQQ/TLT/GLD)", equal_weight),
        ("SPY buy and hold", spy_returns)
    ]:
        print(f"{label:<28} {total_return(r)*100:>7.1f}% {sharpe(r):>8.2f} {max_drawdown(r)*100:>7.1f}%")

    # ── 5. Cumulative return chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, r, color in [
        ("Blended regime portfolio", port_returns, "steelblue"),
        ("Equal weight",             equal_weight,  "coral"),
        ("SPY buy and hold",         spy_returns,   "gray")
    ]:
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, label=label,
                color=color, lw=1.5 if label != "SPY buy and hold" else 1,
                linestyle="-" if label != "SPY buy and hold" else "--")

    ax.set_ylabel("Cumulative return")
    ax.set_title("Regime-aware portfolio vs benchmarks (test set)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig("outputs/portfolio_backtest.png", dpi=150)
    plt.show()

    return port_returns, w_bull, w_bear