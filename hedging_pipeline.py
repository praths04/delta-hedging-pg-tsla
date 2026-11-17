"""
Hedging project -- single-file pipeline.

Consolidated version of the 8-file project (config.py, black_scholes.py,
data_utils.py, hedging_engine.py, get_data.py, run_q1_vol_premium.py,
run_q2_regime.py, plot_results.py) for easy sharing/archiving. The section
headers below mark where each original file's code lives; nothing about the
logic was changed, just merged into one file with shared imports/constants.

Research questions (BU.232.650 final project -- PG vs TSLA hedging):
  Q1 (volatility risk premium): Does implied vol persistently run above
      realized vol for PG and TSLA, such that delta-hedging a sold ATM call
      is profitable on average -- and does the premium differ between TSLA
      (high-vol growth) and PG (defensive)?
  Q2 (regime dependence): Does hedging error behave differently across
      market regimes -- 2023 calm baseline, 2022 rate-hike stress, and the
      Jan 2025 DeepSeek-selloff AI shock -- and is that regime sensitivity
      more pronounced for TSLA than PG?

Methodology: a Black-Scholes delta-hedging simulator rolls ~30-day
at-the-money call options across the sample, tracking self-financing hedge
P&L (sell the option, hold Delta shares, rebalance daily, settle at expiry).
Q1 varies the volatility input used to hedge (realized vs implied). Q2 keeps
the vol input fixed (realized) and varies which time window the engine runs
over.

Data source: Bloomberg Terminal, HIVG function (30-day tenor, Classical
model, 252 annualization), exported as PG_Options_Data_2022_2025.xlsx and
Tesla_Options_Data_2022_2025.xlsx -- each file has Date, Price, and Implied
Vol (in percent) already aligned on the same dates, so no separate Yahoo
Finance pull is needed for either question; the Price column feeds both the
realized-vol calculation and the hedge simulation directly.

Run order:
    pip install -r requirements.txt openpyxl
    python hedging_pipeline.py
(with PG_Options_Data_2022_2025.xlsx and Tesla_Options_Data_2022_2025.xlsx
in the same folder as this script)
This loads both tickers' data, runs Q1, runs Q2, and saves the report
figures to figures/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


# =============================================================================
# config.py -- shared settings
# =============================================================================

TICKERS = {
    "PG": {"dividend_yield": 0.025},   # P&G pays a real dividend, roughly ~2.5%/yr
    "TSLA": {"dividend_yield": 0.0},   # Tesla pays no dividend
}

# Bloomberg export file per ticker -- Date, Price, Implied Vol (percent),
# already aligned on the same dates. Update these paths if you rename the
# files or move them into a subfolder.
DATA_FILES = {
    "PG": "PG_Options_Data_2022_2025.xlsx",
    "TSLA": "Tesla_Options_Data_2022_2025.xlsx",
}

# Actual coverage of the Bloomberg export -- both tickers run Jan 3, 2022
# through Apr 30, 2025. This covers all three Q2 regime windows below and
# gives Q1 a healthy non-overlapping sample. Update if you re-export with a
# different range.
IV_SAMPLE_START = "2022-01-01"
IV_SAMPLE_END = "2025-04-30"

# ~3.5 years gives ~40+ non-overlapping ~30-day cycles per stock, which is
# enough for a clean independent-observations sample -- no need for the
# overlapping-cycle workaround anymore. Setting step_days = maturity_days
# (via ROLL_STEP_DAYS = None) rolls strictly back-to-back options. If you
# ever have to fall back to a shorter IV window again, set ROLL_STEP_DAYS to
# something less than MATURITY_TRADING_DAYS (e.g. 5) to get overlapping
# cycles instead, and note the correlated-observations caveat in the report.
ROLL_STEP_DAYS = None

# A single constant risk-free rate for simplicity (reasonable for a course
# project). If you want to be more precise, swap this for the 3-month T-bill
# yield (^IRX on Yahoo Finance) resampled to match your dates.
RISK_FREE_RATE = 0.02

# ~30 calendar days ~= 21 trading days; used both for the rolling option
# maturity and the realized-vol lookback window (so both are "1-month" vol)
MATURITY_TRADING_DAYS = 21

# Regime windows for research question 2. 2020 COVID isn't usable since it
# falls outside Bloomberg's 5-year HIVG window.
REGIME_WINDOWS = {
    "2023_calm": ("2023-01-01", "2023-12-31"),
    "2022_rate_hike_stress": ("2022-01-01", "2022-12-31"),
    "2025_ai_shock": ("2025-01-01", "2025-04-30"),  # DeepSeek selloff (Jan 27, 2025) + aftermath
}


# =============================================================================
# black_scholes.py -- Black-Scholes price + Delta
# =============================================================================

def bs_price(S, K, T, r, q, sigma, option_type="call"):
    """
    Black-Scholes price of a European option.

    S : spot price
    K : strike
    T : time to maturity in YEARS (e.g. 30/365)
    r : continuously compounded risk-free rate (annualized)
    q : continuously compounded dividend yield (annualized)
    sigma : volatility input (annualized) -- this is the "hedging vol" you plug
            in, so you can swap historical vol vs implied vol here
    option_type : "call" or "put"
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S, K, T, r, q, sigma, option_type="call"):
    """Delta (spot derivative) of a European option -- the hedge ratio."""
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return -np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# =============================================================================
# data_utils.py -- load Bloomberg export, realized vol, alignment
# =============================================================================

TRADING_DAYS_PER_YEAR = 252

_DATA_CACHE = {}


def load_ticker_data(ticker: str) -> pd.DataFrame:
    """
    Load one ticker's Bloomberg export (DATA_FILES[ticker]): Date, Price,
    Implied Vol (percent). Returns a DataFrame indexed by Date with columns
    'price', 'implied_vol' (decimal), and 'realized_vol' (trailing 21-day,
    annualized, computed from 'price' -- matches the ~30-day option maturity).
    Cached per ticker since several functions below need this repeatedly.
    """
    if ticker in _DATA_CACHE:
        return _DATA_CACHE[ticker]

    df = pd.read_excel(DATA_FILES[ticker])
    df = df.dropna(subset=["Price", "Implied Vol"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    out = pd.DataFrame(index=df.index)
    out["price"] = df["Price"].astype(float)
    out["implied_vol"] = df["Implied Vol"].astype(float) / 100.0  # Bloomberg exports percent
    out["realized_vol"] = realized_vol(out["price"], window=MATURITY_TRADING_DAYS)

    _DATA_CACHE[ticker] = out
    return out


def realized_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    """
    Trailing annualized realized volatility from daily log returns.
    window=21 trading days ~= 1 month, matching the ~30-day option maturity.
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def align_series(*series_list: pd.Series) -> pd.DataFrame:
    """Inner-join a set of date-indexed Series into one aligned DataFrame."""
    out = pd.concat(series_list, axis=1, join="inner")
    return out.dropna()


# =============================================================================
# hedging_engine.py -- core delta-hedging simulator
#
# This is the ONE shared engine both research questions run on top of:
#   - Q1 (vol risk premium) varies the *volatility input* (historical vs implied).
#   - Q2 (regime dependence) varies the *time window* the engine is run over.
#
# Logic (self-financing replication portfolio):
#   1. At t=0, sell the option for its BS price (using the vol you're testing)
#      and buy Delta_0 shares. Cash = premium - Delta_0 * S_0.
#   2. Each day: cash grows at the risk-free rate, then re-hedge: cost of
#      buying/selling the change in Delta is paid out of cash.
#   3. At expiry: settle the option's payoff, liquidate the stock position.
#   4. Final portfolio value = leftover cash + terminal stock value - payoff.
#      If hedging were perfect and continuous, this would be ~0. The *size and
#      variance* of this leftover is your "hedging error" / "hedging P&L".
# =============================================================================

def simulate_single_option_hedge(
    prices: np.ndarray,
    hedge_vols: np.ndarray,
    K: float,
    r: float,
    q: float,
    option_type: str = "call",
):
    """
    Delta-hedge ONE option across its life.

    prices     : array of daily spot prices, length n (from inception to expiry)
    hedge_vols : array of the "hedging vol" input used on each day, length n
    K          : strike, fixed at inception (normally set = prices[0], i.e. ATM)
    r, q       : annualized risk-free rate and dividend yield (can be scalars)
    option_type: "call" or "put"

    Returns a dict with the premium collected, final hedging P&L, and the
    day-by-day portfolio path (handy for plotting / sanity checking).
    """
    n = len(prices)
    assert len(hedge_vols) == n, "prices and hedge_vols must be the same length"

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    # time-to-maturity counts down from T to 0 across the n days
    T_path = np.array([(n - 1 - i) * dt for i in range(n)])

    premium = bs_price(prices[0], K, T_path[0], r, q, hedge_vols[0], option_type)
    delta_prev = bs_delta(prices[0], K, T_path[0], r, q, hedge_vols[0], option_type)

    cash = premium - delta_prev * prices[0]
    portfolio_value = [cash + delta_prev * prices[0]]  # == premium at t=0

    for i in range(1, n):
        cash *= np.exp(r * dt)  # cash account accrues interest overnight
        S_t, T_t = prices[i], T_path[i]

        if T_t > 1e-9:
            delta_new = bs_delta(S_t, K, T_t, r, q, hedge_vols[i], option_type)
        else:
            delta_new = 1.0 if (option_type == "call" and S_t > K) else 0.0

        cash -= (delta_new - delta_prev) * S_t  # cost of rebalancing
        delta_prev = delta_new
        portfolio_value.append(cash + delta_prev * S_t)

    S_T = prices[-1]
    payoff = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
    final_hedge_pnl = cash + delta_prev * S_T - payoff

    return {
        "premium": premium,
        "hedge_pnl": final_hedge_pnl,
        "hedge_pnl_pct_of_premium": final_hedge_pnl / premium if premium > 0 else np.nan,
        "portfolio_path": portfolio_value,
    }


def roll_hedging_strategy(
    price_series: pd.Series,
    hedge_vol_series: pd.Series,
    r: float,
    q: float,
    option_type: str = "call",
    maturity_days: int = 21,  # ~30 calendar days = ~21 trading days
    step_days: int = None,
):
    """
    Chain together many ~30-day ATM options over a sample.

    step_days controls how far apart consecutive option cycles start:
      - step_days = maturity_days (the default, i.e. step_days=None) gives
        the original "roll monthly" behaviour: each option starts right when
        the previous one expires, so cycles never overlap.
      - step_days < maturity_days gives OVERLAPPING cycles (e.g. step_days=5
        starts a new ~30-day hedge every week) -- trades independence of
        observations for a usable sample size when the sample is short; note
        in your report that overlapping observations are correlated, not
        independent draws, if you ever use this.

    price_series, hedge_vol_series : pandas Series indexed by date, aligned.
    Returns a DataFrame, one row per option cycle, with start/end dates,
    the strike used, premium collected, and hedge P&L.
    """
    if step_days is None:
        step_days = maturity_days

    dates = price_series.index
    results = []
    start_idx = 0
    n = len(price_series)

    while start_idx + maturity_days < n:
        end_idx = start_idx + maturity_days
        window_prices = price_series.iloc[start_idx : end_idx + 1].values
        window_vols = hedge_vol_series.iloc[start_idx : end_idx + 1].values

        if np.any(pd.isna(window_vols)):
            # not enough history yet to compute a vol input (e.g. realized vol
            # warm-up period) -- skip this cycle
            start_idx += step_days
            continue

        K = window_prices[0]  # ATM at inception
        out = simulate_single_option_hedge(window_prices, window_vols, K, r, q, option_type)

        results.append(
            {
                "start_date": dates[start_idx],
                "end_date": dates[end_idx],
                "K": K,
                "premium": out["premium"],
                "hedge_pnl": out["hedge_pnl"],
                "hedge_pnl_pct_of_premium": out["hedge_pnl_pct_of_premium"],
            }
        )
        start_idx += step_days  # next cycle starts here -- may overlap the current one

    return pd.DataFrame(results)


# =============================================================================
# get_data.py -- Step 1: load and sanity-check both tickers' Bloomberg export
# =============================================================================

def load_and_check_data():
    for ticker in TICKERS:
        df = load_ticker_data(ticker)
        print(f"{ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
        print(df.head())
        print()
    print("Done. Both tickers loaded straight from their Bloomberg export "
          "(no separate Yahoo Finance pull or IV CSV needed).")


# =============================================================================
# run_q1_vol_premium.py -- Research Question 1: volatility risk premium
#
# Does implied vol persistently run above realized vol for PG and TSLA, such
# that delta-hedging a sold option is profitable on average? Tested by
# running the SAME rolling ~30-day ATM call hedge twice per stock:
#   (a) hedged using trailing realized (historical) vol
#   (b) hedged using the market's implied vol
# and comparing the distribution of hedge P&L between the two.
#
# Intuition: if you *sell* the option and delta-hedge with realized vol (the
# vol that actually happens), your hedge P&L should be ~0 on average (perfect
# replication, up to discretization error). If instead you collect a premium
# priced off implied vol but the world only realizes the (usually lower)
# realized vol, you pocket the difference -- that's the risk premium.
# =============================================================================

def run_q1_for_ticker(ticker: str):
    div_yield = TICKERS[ticker]["dividend_yield"]
    df = load_ticker_data(ticker).loc[IV_SAMPLE_START:IV_SAMPLE_END]
    df = df.dropna(subset=["realized_vol", "implied_vol"])

    results_hist = roll_hedging_strategy(
        df["price"], df["realized_vol"],
        r=RISK_FREE_RATE, q=div_yield,
        maturity_days=MATURITY_TRADING_DAYS,
        step_days=ROLL_STEP_DAYS,
    )
    results_hist["vol_input"] = "realized"

    results_iv = roll_hedging_strategy(
        df["price"], df["implied_vol"],
        r=RISK_FREE_RATE, q=div_yield,
        maturity_days=MATURITY_TRADING_DAYS,
        step_days=ROLL_STEP_DAYS,
    )
    results_iv["vol_input"] = "implied"

    return pd.concat([results_hist, results_iv], ignore_index=True)


def summarize_q1(results: pd.DataFrame, ticker: str):
    print(f"\n=== {ticker}: hedge P&L as % of premium, by vol input ===")
    summary = results.groupby("vol_input")["hedge_pnl_pct_of_premium"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    print(summary)

    if set(["realized", "implied"]).issubset(results["vol_input"].unique()):
        a = results.loc[results.vol_input == "realized", "hedge_pnl_pct_of_premium"]
        b = results.loc[results.vol_input == "implied", "hedge_pnl_pct_of_premium"]
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        print(f"Welch t-test (realized vs implied hedge P&L): t={t_stat:.3f}, p={p_val:.4f}")
        overlapping = ROLL_STEP_DAYS is not None and ROLL_STEP_DAYS < MATURITY_TRADING_DAYS
        if overlapping:
            print(
                "Note: cycles overlap (step_days < maturity_days), so these observations "
                "are correlated, not independent -- treat the p-value as indicative, not exact."
            )
        else:
            print("Note: cycles are non-overlapping (step_days = maturity_days).")


def run_q1_vol_premium():
    all_results = {}
    for ticker in TICKERS:
        res = run_q1_for_ticker(ticker)
        all_results[ticker] = res
        res.to_csv(f"q1_{ticker}_results.csv", index=False)
        summarize_q1(res, ticker)
    print("\nSaved per-ticker results to q1_<TICKER>_results.csv")
    return all_results


# =============================================================================
# run_q2_regime.py -- Research Question 2: regime dependence
#
# Does hedging error behave differently in calm markets (2023) vs volatile
# ones (2022 rate-hike stress, 2025 AI shock), and is that regime sensitivity
# more pronounced for TSLA (volatile) than PG (defensive)?
#
# Reuses the exact same rolling delta-hedge engine as Q1 -- the only thing
# that changes is WHICH dates it runs over. By default this hedges using
# realized (historical) vol, since that isolates "how hard is this stock to
# hedge in this regime" from the implied-vol risk-premium question in Q1.
# =============================================================================

Q2_VOL_INPUT = "realized"  # "realized" or "implied" -- both columns are on
                            # load_ticker_data()'s output, so switching this
                            # to "implied" as a robustness check is a one-line change


def run_q2_for_ticker_and_regime(ticker: str, regime_name: str, start: str, end: str):
    div_yield = TICKERS[ticker]["dividend_yield"]
    df = load_ticker_data(ticker)
    vol_col = "realized_vol" if Q2_VOL_INPUT == "realized" else "implied_vol"
    df = df.dropna(subset=[vol_col])

    window = df.loc[start:end]
    if window.empty:
        print(f"[{ticker} / {regime_name}] no data in this window -- check dates / data availability.")
        return pd.DataFrame()

    results = roll_hedging_strategy(
        window["price"], window[vol_col],
        r=RISK_FREE_RATE, q=div_yield,
        maturity_days=MATURITY_TRADING_DAYS,
    )
    results["ticker"] = ticker
    results["regime"] = regime_name
    return results


def run_q2_regime():
    all_results = []
    for ticker in TICKERS:
        for regime_name, (start, end) in REGIME_WINDOWS.items():
            res = run_q2_for_ticker_and_regime(ticker, regime_name, start, end)
            all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("q2_regime_results.csv", index=False)

    print("\n=== Hedge P&L (% of premium) by ticker and regime ===")
    print(
        combined.groupby(["ticker", "regime"])["hedge_pnl_pct_of_premium"].agg(
            ["count", "mean", "std"]
        )
    )
    print("\nSaved combined results to q2_regime_results.csv")
    return combined


# =============================================================================
# plot_results.py -- figures for the report/slide deck
#
# Reads q1_{ticker}_results.csv and q2_regime_results.csv (written above),
# plus recomputes the aligned price/vol series for the vol time-series chart.
# Saves four PNGs to a local figures/ folder.
# =============================================================================

FIG_DIR = "figures"

# Fixed color per entity, assigned once and reused everywhere -- never
# recycled or reassigned based on rank/filter state.
TICKER_COLOR = {"PG": "#2a78d6", "TSLA": "#008300"}          # blue / green
VOL_INPUT_COLOR = {"realized": "#2a78d6", "implied": "#eb6834"}  # blue / orange

REGIME_ORDER = ["2022_rate_hike_stress", "2023_calm", "2025_ai_shock"]
REGIME_LABEL = {
    "2022_rate_hike_stress": "2022\n(rate-hike stress)",
    "2023_calm": "2023\n(calm baseline)",
    "2025_ai_shock": "2025\n(AI shock, Jan-Apr)",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#888888",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def fig1_vol_premium_timeseries():
    """IV vs trailing realized vol over time, one panel per ticker -- the
    visual version of Q1's premise: does implied vol persistently sit above
    realized vol?"""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    for ax, ticker in zip(axes, TICKERS):
        df = load_ticker_data(ticker).loc[IV_SAMPLE_START:IV_SAMPLE_END]
        aligned = df.rename(columns={"realized_vol": "realized", "implied_vol": "implied"}).dropna(
            subset=["realized", "implied"]
        )

        ax.plot(aligned.index, aligned["implied"] * 100, label="Implied vol (IV30)",
                color=VOL_INPUT_COLOR["implied"], linewidth=1.3)
        ax.plot(aligned.index, aligned["realized"] * 100, label="Realized vol (21d trailing)",
                color=VOL_INPUT_COLOR["realized"], linewidth=1.3)
        ax.fill_between(
            aligned.index, aligned["realized"] * 100, aligned["implied"] * 100,
            where=(aligned["implied"] >= aligned["realized"]),
            color=VOL_INPUT_COLOR["implied"], alpha=0.10, interpolate=True,
        )
        ax.set_title(f"{ticker}: implied vs realized volatility", loc="left", fontsize=12)
        ax.set_ylabel("Annualized vol (%)")
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out = f"{FIG_DIR}/fig1_vol_premium_timeseries.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig2_q1_hedge_pnl_boxplot():
    """Q1 result: distribution of hedge P&L (% of premium), realized vs
    implied vol input, one panel per ticker."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    order = ["realized", "implied"]
    for ax, ticker in zip(axes, TICKERS):
        df = pd.read_csv(f"q1_{ticker}_results.csv")
        data = [df.loc[df.vol_input == v, "hedge_pnl_pct_of_premium"].dropna() for v in order]
        bp = ax.boxplot(data, tick_labels=[v.capitalize() for v in order],
                         patch_artist=True, widths=0.5, medianprops=dict(color="black"))
        for patch, v in zip(bp["boxes"], order):
            patch.set_facecolor(VOL_INPUT_COLOR[v])
            patch.set_alpha(0.55)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(ticker, loc="left", fontsize=12)
        ax.set_xlabel("Hedging vol input")
    axes[0].set_ylabel("Hedge P&L (% of premium)")
    fig.suptitle("Q1 -- Hedge P&L distribution: realized vs implied vol input", fontsize=12)
    fig.tight_layout()
    out = f"{FIG_DIR}/fig2_q1_hedge_pnl_boxplot.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig3_q2_regime_bars():
    """Q2 result: mean hedge P&L (% of premium) by regime, grouped by
    ticker, with +/-1 standard-error bars."""
    df = pd.read_csv("q2_regime_results.csv")
    stats_df = (
        df.groupby(["ticker", "regime"])["hedge_pnl_pct_of_premium"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats_df["se"] = stats_df["std"] / np.sqrt(stats_df["count"])

    tickers = list(TICKERS)
    x = np.arange(len(REGIME_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ticker in enumerate(tickers):
        sub = stats_df[stats_df.ticker == ticker].set_index("regime").reindex(REGIME_ORDER)
        offset = (i - (len(tickers) - 1) / 2) * width
        ax.bar(x + offset, sub["mean"], width, yerr=sub["se"], capsize=4,
               label=ticker, color=TICKER_COLOR[ticker], alpha=0.85)

    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL[r] for r in REGIME_ORDER])
    ax.set_ylabel("Mean hedge P&L (% of premium)")
    ax.set_title("Q2 -- Hedge P&L by regime and ticker (±1 SE)", loc="left", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = f"{FIG_DIR}/fig3_q2_regime_bars.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")
    print("Note: the 2025_ai_shock regime has only ~3 non-overlapping cycles "
          "per ticker (4-month window) -- its bar/error bar is noisy, say so in the report.")


def fig4_q2_regime_boxplot():
    """Q2 supplementary: full distribution (not just the mean) of hedge P&L
    by regime, one panel per ticker -- shows whether variance widens in
    stress regimes, which fig3's error bars can only hint at."""
    df = pd.read_csv("q2_regime_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, ticker in zip(axes, TICKERS):
        sub = df[df.ticker == ticker]
        data = [sub.loc[sub.regime == r, "hedge_pnl_pct_of_premium"].dropna() for r in REGIME_ORDER]
        bp = ax.boxplot(data, tick_labels=[REGIME_LABEL[r] for r in REGIME_ORDER],
                         patch_artist=True, widths=0.5, medianprops=dict(color="black"))
        for patch in bp["boxes"]:
            patch.set_facecolor(TICKER_COLOR[ticker])
            patch.set_alpha(0.55)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(ticker, loc="left", fontsize=12)
    axes[0].set_ylabel("Hedge P&L (% of premium)")
    fig.suptitle("Q2 -- Hedge P&L distribution by regime", fontsize=12)
    fig.tight_layout()
    out = f"{FIG_DIR}/fig4_q2_regime_boxplot.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def plot_results():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_vol_premium_timeseries()
    fig2_q1_hedge_pnl_boxplot()
    fig3_q2_regime_bars()
    fig4_q2_regime_boxplot()
    print(f"\nAll figures saved to {FIG_DIR}/")


# =============================================================================
# Main -- run the whole pipeline end to end
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1/4: loading Bloomberg export (get_data)")
    print("=" * 70)
    load_and_check_data()

    print("\n" + "=" * 70)
    print("STEP 2/4: Research Question 1 -- volatility risk premium")
    print("=" * 70)
    run_q1_vol_premium()

    print("\n" + "=" * 70)
    print("STEP 3/4: Research Question 2 -- regime dependence")
    print("=" * 70)
    run_q2_regime()

    print("\n" + "=" * 70)
    print("STEP 4/4: plotting figures")
    print("=" * 70)
    plot_results()
