# Delta-Hedging P&G and Tesla: Volatility Risk Premium and Regime Dependence

A Black-Scholes delta-hedging simulation on two very different stocks, Procter & Gamble (defensive, ~2.5% dividend yield) and Tesla (high-volatility growth, no dividend), built for the Continuous Time Finance course (BU.232.650) at Johns Hopkins Carey Business School.

**Tools:** Python (NumPy, pandas, SciPy, Matplotlib), Bloomberg Terminal data

---

## Questions

1. **Volatility risk premium:** Does implied volatility persistently run above realized volatility for PG and TSLA, so that delta-hedging a sold at-the-money call is profitable on average? Does the premium differ between the two stocks?
2. **Regime dependence:** Does hedging error behave differently across a calm baseline (2023), the 2022 rate-hike stress period, and the January 2025 DeepSeek selloff, and is that more pronounced for TSLA than for PG?

We picked PG and TSLA because they sit at opposite ends of most things that matter for hedging: steady versus jumpy prices, low versus high volatility, dividend versus none. If a result holds for both, it is less likely to be an artifact of one type of stock.

## Method

- **Simulator:** sell a ~30-day at-the-money European call, hold the Black-Scholes delta in shares (with a continuous dividend yield), rebalance daily, and settle at expiry. The hedge is self-financing, and P&L is reported as a percentage of the premium received.
- **Rolling cycles:** back-to-back 21-trading-day options across the whole sample, giving about 38 non-overlapping cycles per stock. Cycles with missing volatility data are skipped.
- **Q1:** run the same rolling hedge twice, once with realized volatility and once with implied volatility, and compare the two P&L distributions with a Welch two-sample t-test.
- **Q2:** run the realized-volatility hedge inside each regime window and compare the mean hedge P&L and its standard deviation (the hedging error).
- **Assumptions:** 2% constant risk-free rate, 21-day rolling realized volatility from daily log returns, 252-day annualization.

## Data

Daily closing price and 30-day at-the-money implied volatility for PG and TSLA from January 3, 2022 to April 30, 2025 (about 834 trading days each), from a Bloomberg Terminal (HIVG function, Classical model, 30-day tenor, 252-day annualization).

**The data is not included in this repository.** To rerun the analysis, export `Date`, `Price` and `Implied Vol` (in percent) for each stock and save them next to the notebook as:

- `PG_Options_Data_2022_2025.xlsx`
- `Tesla_Options_Data_2022_2025.xlsx`

## Findings

These are results for two stocks over one sample window, so they should not be read as general rules.

**Q1: no significant volatility risk premium for either stock**

| Ticker | Vol input | Cycles | Mean P&L (% of premium) | Std dev | Welch t | p-value |
|---|---|---|---|---|---|---|
| PG | Implied | 38 | -13.4% | 38.6% | -0.87 | 0.39 |
| PG | Realized | 38 | -23.4% | 60.0% | | |
| TSLA | Implied | 38 | +1.8% | 25.4% | 0.03 | 0.97 |
| TSLA | Realized | 38 | +2.0% | 34.7% | | |

Neither difference is statistically significant. The PG comparison is influenced by a few high-variance cycles, including one (July to August 2024) that lost about $5 against a $2.14 premium. This is consistent with the volatility risk premium being clearest at the index level (S&P 500 / VIX), where diversification smooths out single-stock noise.

**Q2: hedging error is regime-dependent for TSLA but not clearly for PG**

| Ticker | Regime | Cycles | Mean P&L | Hedging error (std dev) |
|---|---|---|---|---|
| PG | 2022 rate-hike stress | 10 | -0.5% | 31.7% |
| PG | 2023 calm | 11 | -20.9% | 46.1% |
| PG | 2025 AI shock | 3 | -35.1% | 39.9% |
| TSLA | 2022 rate-hike stress | 10 | +1.3% | 33.1% |
| TSLA | 2023 calm | 11 | +5.7% | 13.1% |
| TSLA | 2025 AI shock | 3 | +1.1% | 39.5% |

- **TSLA** shows the pattern we expected: hedging error falls to 13.1% in the calm year and rises to 33.1% and 39.5% in the two stress regimes.
- **PG** does not: its error stays elevated in every regime and peaks in 2023, its supposedly calm year. Its average hedge P&L also worsens across regimes (-0.5%, -20.9%, -35.1%).
- The 2025 window has only 3 cycles per stock, so those numbers are directional.
- For PG's 2023 result we looked for documented explanations instead of assuming one: it behaved like a "bond proxy" stock while rates stayed high, and it reported several quarters of falling sales volume. The March 2023 banking crisis and the debt-ceiling standoff did not appear to be strong direct drivers.

The full write-up, with figures, is in `RQ_Findings_PG_TSLA.pdf`.

## Files

| File | What it is |
|---|---|
| `hedging_analysis.ipynb` | Notebook with the full analysis, figures and statistical tests |
| `hedging_pipeline.py` | Same logic as a single script that saves the report figures to `figures/` |
| `RQ_Findings_PG_TSLA.pdf` | Written findings for both research questions |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook hedging_analysis.ipynb
```

Put the two Bloomberg export files in the same folder first. To use the script instead, run `python hedging_pipeline.py`.

## Limitations

- A single constant 2% risk-free rate is used for simplicity.
- Daily rebalancing and no transaction costs, so results are a clean test of the volatility assumption and not a trading strategy.
- About 38 cycles per stock is enough for a basic significance test, but small for strong conclusions, and the 2025 regime has only 3 cycles.

## Authors

Prathiba Reddy and Friedemann Schmidt, MS in Business Analytics & AI, Johns Hopkins Carey Business School
[LinkedIn](https://www.linkedin.com/in/prathiba-reddy-namireddy) | [Portfolio](https://6a38b7b1cee129ca7a613700--comforting-queijadas-d1f251.netlify.app)
