import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn.hmm import GaussianHMM

def run(train_scaled, test_scaled, train, test, prices):

    # ── Prepare observation matrix ────────────────────────────────
    obs_cols = ["spy_ret", "vol_21d"]
    X_train = train_scaled[obs_cols].values
    X_test  = test_scaled[obs_cols].values

    # ── Fit the HMM ───────────────────────────────────────────────
    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=1000,
        tol=1e-4,
        random_state=42
    )
    model.fit(X_train)

    print(f"Converged: {model.monitor_.converged}")
    print(f"Log-likelihood: {model.score(X_train):.2f}")

    # ── Decode regimes ────────────────────────────────────────────
    train_states = model.predict(X_train)
    test_states  = model.predict(X_test)

    # ── Identify bull vs bear ─────────────────────────────────────
    means      = model.means_
    bull_state = int(np.argmax(means[:, 0]))
    bear_state = 1 - bull_state

    label_map     = {bull_state: 0, bear_state: 1}
    train_regimes = pd.Series([label_map[s] for s in train_states], index=train.index)
    test_regimes  = pd.Series([label_map[s] for s in test_states],  index=test.index)

    # ── Inspect learned parameters ────────────────────────────────
    scaler_mean = train_scaled.mean().values
    scaler_std  = train_scaled.std().values
    col_idx     = {c: i for i, c in enumerate(obs_cols)}

    for state, name in [(bull_state, "Bull"), (bear_state, "Bear")]:
        i_ret = col_idx["spy_ret"]
        i_vol = col_idx["vol_21d"]
        raw_mean_ret = means[state, i_ret] * scaler_std[i_ret] + scaler_mean[i_ret]
        raw_mean_vol = means[state, i_vol] * scaler_std[i_vol] + scaler_mean[i_vol]
        print(f"\n{name} regime (state {state}):")
        print(f"  Mean daily return : {raw_mean_ret*100:.3f}%")
        print(f"  Mean ann. vol     : {raw_mean_vol*100:.1f}%")

    print("\nTransition matrix:")
    print(pd.DataFrame(model.transmat_,
                       index=["Bull→", "Bear→"],
                       columns=["→Bull", "→Bear"]).round(3))

    # ── Visualise ─────────────────────────────────────────────────
    all_regimes = pd.concat([train_regimes, test_regimes])
    spy_prices  = prices["spy"].reindex(all_regimes.index)

    plt.style.use('dark_background')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(spy_prices.index, spy_prices.values, color="steelblue", lw=1)
    ax1.fill_between(spy_prices.index, spy_prices.min(), spy_prices.max(),
                     where=(all_regimes == 1),
                     color="salmon", alpha=0.3, label="Bear regime")
    ax1.set_ylabel("S&P 500 price")
    ax1.legend(loc="upper left")
    ax1.set_title("HMM regime detection — S&P 500")

    all_X    = np.vstack([X_train, X_test])
    bear_prob = model.predict_proba(all_X)[:, bear_state]

    ax2.plot(all_regimes.index, bear_prob, color="salmon", lw=1)
    ax2.axhline(0.5, color="gray", linestyle="--", lw=0.8)
    ax2.set_ylabel("P(bear regime)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


    plt.tight_layout()
    plt.savefig("outputs/hmm_regimes.png", dpi=150)
    plt.show()

    return all_regimes, model