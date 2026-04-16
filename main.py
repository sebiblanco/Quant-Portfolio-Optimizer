import os
import sys

# Ensure scripts directory is available for import
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_loader import download_data
from optimizer import run_monte_carlo
from visualizer import plot_efficient_frontier

def main():
    print("=== Portfolio Optimization & Risk Modeling Tool ===\n")
    
    # 1. Configuration
    tickers = ["META", "AMZN", "MELI", "NU"]
    period = "5y"
    num_simulations = 5000
    risk_free_rate = 0.0428 # Current 10-year US Treasury yield
    
    # 2. Extract Data
    print("Step 1: Extracting Financial Data")
    price_data = download_data(tickers, period=period)
    print("\n")
    
    # 3. Optimize Portfolio
    print("Step 2: Running Portfolio Optimization")
    results_df, max_sharpe, min_vol, mean_returns, cov_matrix, frontier_data = run_monte_carlo(
        price_data, 
        num_simulations=num_simulations, 
        risk_free_rate=risk_free_rate
    )
    print("\n")
    
    # 4. Visualize Results
    print("Step 3: Generating Visualizations")
    plot_efficient_frontier(results_df, max_sharpe, min_vol, frontier_data=frontier_data, risk_free_rate=risk_free_rate)
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
