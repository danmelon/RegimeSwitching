import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def run(train, test, prices):

    # ── Prepare data — raw unscaled returns ───────────────────────
    # MS model works on the raw return series directly
    spy_ret_train = train["spy_ret"]
    spy_ret_full  = pd.concat([train["spy_ret"], test["spy_ret"]])

    # ── Fit Markov Switching model ────────────────────────────────
    # switching_variance=True gives each regime its own volatility
    # k_regimes=2 for bull/bear
    ms_model = MarkovRegression(
        spy_ret_train,
        k_regimes=2,
        trend="c",                # constant (intercept) per regime
        switching_variance=True   # separate variance per regime
    )
    ms_result = ms_model.fit(search_reps=20, disp=False)

    print(ms_result.summary())

    # ── Identify bull vs bear by mean return ──────────────────────
    params     = ms_result.params
    # statsmodels orders regimes 0 and 1 — bull = higher mean
    means      = [ms_result.params[f"const[{i}]"] for i in range(2)]
    bull_state = int(np.argmax(means))
    bear_state = 1 - bull_state

    print(f"\nBull regime: state {bull_state}")
    print(f"  Mean daily return : {means[bull_state]*100:.3f}%")
    print(f"  Ann. vol          : {np.sqrt(ms_result.params[f'sigma2[{bull_state}]']*252)*100:.1f}%")
    print(f"\nBear regime: state {bear_state}")
    print(f"  Mean daily return : {means[bear_state]*100:.3f}%")
    print(f"  Ann. vol          : {np.sqrt(ms_result.params[f'sigma2[{bear_state}]']*252)*100:.1f}%")

    # ── Expected regime durations ─────────────────────────────────
    p00 = ms_result.params["p[0->0]"]
    p10 = ms_result.params["p[1->0]"]
    p11 = 1 - p10
    print(f"\nExpected bull duration : {1/(1-p11):.0f} trading days")
    print(f"\nExpected bear duration : {1/(1-p00):.0f} trading days")
    print(f"\nAIC : {ms_result.aic:.2f}")
    print(f"BIC : {ms_result.bic:.2f}")

    # ── Smoothed probabilities ────────────────────────────────────
    # Smoothed = P(regime | entire dataset) — uses future info, better for analysis
    # Filtered = P(regime | data up to t)   — realistic, no lookahead
    smoothed = ms_result.smoothed_marginal_probabilities
    filtered = ms_result.filtered_marginal_probabilities

    bear_smoothed = smoothed.iloc[:, bear_state]
    bear_filtered = filtered.iloc[:, bear_state]

    # ── Visualise ─────────────────────────────────────────────────
    spy_train_prices = prices["spy"].reindex(train.index)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Price with smoothed regime shading
    ax1.plot(spy_train_prices.index, spy_train_prices.values, color="steelblue", lw=1)
    ax1.fill_between(spy_train_prices.index, spy_train_prices.min(), spy_train_prices.max(),
                     where=(bear_smoothed > 0.5),
                     color="salmon", alpha=0.3, label="Bear regime")
    ax1.set_ylabel("S&P 500 price")
    ax1.legend(loc="upper left")
    ax1.set_title("Markov Switching model — regime detection (train set)")

    # Smoothed probability
    ax2.plot(bear_smoothed.index, bear_smoothed.values, color="salmon", lw=1)
    ax2.axhline(0.5, color="gray", linestyle="--", lw=0.8)
    ax2.set_ylabel("P(bear) smoothed")

    # Filtered probability — more realistic, no future info
    ax3.plot(bear_filtered.index, bear_filtered.values, color="coral", lw=1)
    ax3.axhline(0.5, color="gray", linestyle="--", lw=0.8)
    ax3.set_ylabel("P(bear) filtered")
    ax3.set_xlabel("Date")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("outputs/ms_regimes.png", dpi=150)
    plt.show()

    return ms_result, bull_state, bear_state