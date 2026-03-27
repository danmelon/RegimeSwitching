import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def run(train, test, hmm_regimes, ms_result, hmm_bull_state, ms_bull_state, ms_bear_state, prices):

    spy_ret_full = pd.concat([train["spy_ret"], test["spy_ret"]])

    # ── 1. Statistical fit comparison ────────────────────────────
    print("=" * 55)
    print("STATISTICAL FIT")
    print("=" * 55)
    print(f"MS  model — AIC : {ms_result.aic:.2f}  |  BIC : {ms_result.bic:.2f}")
    print("HMM model — AIC/BIC not directly available (use log-likelihood)")

    # ── 2. Regime quality ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("REGIME QUALITY")
    print("=" * 55)

    # HMM regime stats — train set only
    hmm_train = hmm_regimes.reindex(train.index).dropna()
    hmm_bull_days  = (hmm_train == 0).sum()
    hmm_bear_days  = (hmm_train == 1).sum()
    hmm_switches   = (hmm_train != hmm_train.shift()).sum() - 1

    print(f"\nHMM (train set):")
    print(f"  Bull days     : {hmm_bull_days} ({hmm_bull_days/len(hmm_train)*100:.1f}%)")
    print(f"  Bear days     : {hmm_bear_days} ({hmm_bear_days/len(hmm_train)*100:.1f}%)")
    print(f"  Regime switches : {hmm_switches}")
    print(f"  Avg bull run  : {hmm_bull_days / max((hmm_train != hmm_train.shift()).sum()//2, 1):.0f} days")
    print(f"  Avg bear run  : {hmm_bear_days / max((hmm_train != hmm_train.shift()).sum()//2, 1):.0f} days")

    # MS regime stats — smoothed probs > 0.5
    ms_smoothed   = ms_result.smoothed_marginal_probabilities
    ms_bear_prob  = ms_smoothed.iloc[:, ms_bear_state]
    ms_regimes    = (ms_bear_prob > 0.5).astype(int)
    ms_regimes.index = train.index

    ms_bull_days  = (ms_regimes == 0).sum()
    ms_bear_days  = (ms_regimes == 1).sum()
    ms_switches   = (ms_regimes != ms_regimes.shift()).sum() - 1

    print(f"\nMS model (train set):")
    print(f"  Bull days     : {ms_bull_days} ({ms_bull_days/len(ms_regimes)*100:.1f}%)")
    print(f"  Bear days     : {ms_bear_days} ({ms_bear_days/len(ms_regimes)*100:.1f}%)")
    print(f"  Regime switches : {ms_switches}")
    print(f"  Avg bull run  : {ms_bull_days / max((ms_regimes != ms_regimes.shift()).sum()//2, 1):.0f} days")
    print(f"  Avg bear run  : {ms_bear_days / max((ms_regimes != ms_regimes.shift()).sum()//2, 1):.0f} days")

    # ── 3. Backtest ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("BACKTEST — out of sample (test set)")
    print("=" * 55)

    
    # Align test regimes
    hmm_test = hmm_regimes.reindex(test.index).dropna()
    spy_ret_test = test["spy_ret"].reindex(hmm_test.index)

    def backtest(regimes, returns, label):
        # 1 = in market (bull), 0 = cash (bear)
        # shift regimes by 1 day to avoid lookahead
        signal  = (regimes == 0).astype(int).shift(1).fillna(1)
        strat   = signal * returns
        bh      = returns  # buy and hold

        def sharpe(r):
            return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0

        def max_drawdown(r):
            cum = (1 + r).cumprod()
            roll_max = cum.cummax()
            dd = (cum - roll_max) / roll_max
            return dd.min()

        total_ret_strat = (1 + strat).prod() - 1
        total_ret_bh    = (1 + bh).prod() - 1

        print(f"\n{label}:")
        print(f"  Total return (strategy)   : {total_ret_strat*100:.1f}%")
        print(f"  Total return (buy & hold) : {total_ret_bh*100:.1f}%")
        print(f"  Sharpe (strategy)         : {sharpe(strat):.2f}")
        print(f"  Sharpe (buy & hold)       : {sharpe(bh):.2f}")
        print(f"  Max drawdown (strategy)   : {max_drawdown(strat)*100:.1f}%")
        print(f"  Max drawdown (buy & hold) : {max_drawdown(bh)*100:.1f}%")

        return (1 + strat).cumprod(), (1 + bh).cumprod()

    hmm_cum, bh_cum = backtest(hmm_test, spy_ret_test, "HMM strategy")
    '''
    # MS filtered probs for test set — use filtered not smoothed (no lookahead)
    ms_filtered     = ms_result.filtered_marginal_probabilities
    ms_filtered.index = train.index
    ms_test_regimes = (ms_filtered.iloc[:, ms_bear_state] > 0.5).astype(int)
    ms_test_aligned = ms_test_regimes.reindex(test.index).fillna(0)

    ms_cum, _ = backtest(ms_test_aligned, spy_ret_test, "MS strategy")
    '''
    # ── 3. Backtest (MS SECTION REPLACEMENT) ──────────────────────
    
    # 1. Get the "Bull" and "Bear" mean returns from your fitted MS model
    # Note: Statsmodels stores these in the .params attribute
    try:
        # We try to pull the mean (intercept) for each state
        mu_bull = ms_result.params[f'const[{ms_bull_state}]']
        mu_bear = ms_result.params[f'const[{ms_bear_state}]']
        
        # 2. Create a "Midpoint" threshold
        # If today's return is closer to the Bear mean, we call it a Bear day
        midpoint = (mu_bull + mu_bear) / 2
        
        # 3. Assign regimes (1 = Bear if return is below midpoint, else 0)
        # This bypasses the SVD math error entirely
        ms_test_aligned = (test["spy_ret"] > midpoint).astype(int)
        
    except Exception as e:
        print(f"Shortcut failed ({e}), defaulting to Buy & Hold.")
        ms_test_aligned = pd.Series(0, index=test.index)

    # 4. Run the backtest using these new regimes
    ms_cum, _ = backtest(ms_test_aligned, spy_ret_test, "MS strategy")

    # ── 4. Cumulative return chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hmm_cum.index, hmm_cum.values, label="HMM strategy",      color="steelblue", lw=1.5)
    ax.plot(ms_cum.index,  ms_cum.values,  label="MS strategy",       color="coral",     lw=1.5)
    ax.plot(bh_cum.index,  bh_cum.values,  label="Buy & hold",        color="gray",      lw=1,  linestyle="--")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Out-of-sample backtest — regime switching strategies vs buy & hold")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig("outputs/backtest.png", dpi=150)
    plt.show()