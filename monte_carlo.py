import numpy as np
import matplotlib.pyplot as plt

def run_simulation(strat_returns, bh_returns, label="Strategy", iterations=1000):
    """
    Performs a bootstrap Monte Carlo simulation on strategy returns.
    """
    print("\n" + "=" * 55)
    print(f"MONTE CARLO: {label} ({iterations} iterations)")
    print("=" * 55)
    
    # Calculate actual B&H total return for comparison
    bh_total = (1 + bh_returns).prod() - 1
    
    mc_results = []
    for _ in range(iterations):
        # Bootstrap: Sample returns with replacement
        sample = np.random.choice(strat_returns, size=len(strat_returns), replace=True)
        mc_results.append((1 + sample).prod() - 1)
    
    mc_results = np.array(mc_results)
    
    # Stats
    win_rate = (mc_results > bh_total).mean() * 100
    p5 = np.percentile(mc_results, 5) * 100
    p95 = np.percentile(mc_results, 95) * 100
    
    print(f"  Mean Return      : {mc_results.mean()*100:.1f}%")
    print(f"  5th Percentile   : {p5:.1f}% (Worst Case)")
    print(f"  95th Percentile  : {p95:.1f}% (Best Case)")
    print(f"  Prob. Beat B&H   : {win_rate:.1f}%")

    return mc_results, bh_total