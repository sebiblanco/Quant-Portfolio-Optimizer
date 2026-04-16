# Quantitative Portfolio Optimization & Risk Modeling

## Project Overview
As a Data Science graduate and CFA Level 1 pass, I built this Python-based Mean-Variance Optimization tool to mathematically construct an optimal equity portfolio. The objective is to maximize risk-adjusted returns (Sharpe Ratio) across a highly volatile, cross-border tech and fintech portfolio (META, AMZN, MELI, NU).

## Methodology & Finance Logic
* **Data Ingestion:** Automated extraction of 5 years of historical Adjusted Close price data using the `yfinance` API to accurately capture stock splits and dividend adjustments.
* **Statistical Simulation:** Executed a 5,000-iteration Monte Carlo simulation to plot random asset weightings, visualizing the underlying risk/return distribution.
* **Deterministic Optimization:** Leveraged `scipy.optimize` (Sequential Least Squares Programming) to minimize volatility for target returns, explicitly calculating the continuous **Efficient Frontier** (Markowitz Bullet).
* **Capital Allocation Line (CAL):** Incorporated the current 10-Year U.S. Treasury Yield (4.28%) as the risk-free rate to calculate the tangency portfolio, proving the mathematical superiority of blending the optimal risky portfolio with a risk-free asset.

## Optimization Results & Visualization
The algorithm successfully identified the Global Minimum Volatility Portfolio and the Maximum Sharpe Ratio Portfolio, heavily rotating asset classes based on strict mathematical constraints rather than historical bias.

![Efficient Frontier Visualization](efficient_frontier.ppg)

### Optimal Asset Allocations:
**Maximum Sharpe Ratio Portfolio (Tangency):**
* META: 92.1%
* AMZN: 2.1%
* MELI: 4.4%
* NU: 1.4%

**Minimum Volatility Portfolio:**
* AMZN: 57.2%
* NU: 10.5%
* MELI: 14.4%
* META: 17.9%

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, SciPy, Matplotlib, Seaborn, yFinance
