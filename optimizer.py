import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, trading_days=252):
    """
    Calculates the annualized expected return and volatility for a given setup of portfolio weights.
    """
    # Annualized return
    returns = np.sum(mean_returns * weights) * trading_days
    # Annualized volatility
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    return returns, std

def calculate_efficient_frontier(mean_returns, cov_matrix, return_range, trading_days=252):
    """Calculates the minimum volatility for a given target return."""
    frontier_volatility = []
    num_assets = len(mean_returns)
    
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
        
    def portfolio_return(weights, mean_returns):
        return np.sum(mean_returns * weights) * trading_days

    for target in return_range:
        constraints = (
            {'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]
        
        opt = minimize(portfolio_volatility, initial_weights, args=(cov_matrix,), 
                       method='SLSQP', bounds=bounds, constraints=constraints)
        if opt.success:
            frontier_volatility.append(opt.fun)
        else:
            frontier_volatility.append(np.nan)
            
    return np.array(frontier_volatility)

def run_monte_carlo(price_data, num_simulations=5000, risk_free_rate=0.02):
    """
    Executes a Monte Carlo simulation for portfolio optimization.
    
    Args:
        price_data (pd.DataFrame): DataFrame containing daily adjusted close prices for the assets.
        num_simulations (int): Number of portfolios to simulate.
        risk_free_rate (float): The benchmark risk-free rate to calculate the Sharpe ratio.
        
    Returns:
        tuple: (results_df, max_sharpe_portfolio, min_vol_portfolio, mean_returns, cov_matrix)
    """
    print("Calculating daily logarithmic returns and covariance matrix...")
    
    # Calculate daily log returns
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    num_assets = len(price_data.columns)
    tickers = price_data.columns.tolist()
    
    print(f"Running {num_simulations} Monte Carlo simulations...")
    
    # Pre-allocate arrays storing the simulation outputs
    all_weights = np.zeros((num_simulations, num_assets))
    ret_arr = np.zeros(num_simulations)
    vol_arr = np.zeros(num_simulations)
    sharpe_arr = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Save weights
        all_weights[i,:] = weights
        
        # Expected return and volatility
        p_returns, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        
        # Store results
        ret_arr[i] = p_returns
        vol_arr[i] = p_std
        # Sharpe ratio
        sharpe_arr[i] = (p_returns - risk_free_rate) / p_std
        
    # Store everything in a master DataFrame
    sim_data = {
        'Return': ret_arr,
        'Volatility': vol_arr,
        'Sharpe Ratio': sharpe_arr
    }
    
    for counter, ticker in enumerate(tickers):
        sim_data[f'{ticker} Weight'] = all_weights[:, counter]
        
    results_df = pd.DataFrame(sim_data)
    
    # Identify the optimal portfolios
    max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
    min_vol_idx = results_df['Volatility'].idxmin()
    
    max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]
    min_vol_portfolio = results_df.iloc[min_vol_idx]
    
    def print_portfolio(title, portfolio):
        print(f"\n--- {title} ---")
        print(f"Expected Return: {portfolio['Return']*100:.2f}%")
        print(f"Volatility:      {portfolio['Volatility']*100:.2f}%")
        print(f"Sharpe Ratio:    {portfolio['Sharpe Ratio']:.4f}")
        print("Allocation:")
        for ticker in tickers:
            print(f"  {ticker}: {portfolio[f'{ticker} Weight']*100:.2f}%")

    print("Optimization Complete.")
    print_portfolio("Maximum Sharpe Ratio Portfolio", max_sharpe_portfolio)
    print_portfolio("Minimum Volatility Portfolio", min_vol_portfolio)
    
    print("Calculating Continuous Efficient Frontier...")
    target_returns = np.linspace(results_df['Return'].min(), results_df['Return'].max(), 50)
    frontier_vols = calculate_efficient_frontier(mean_returns, cov_matrix, target_returns)
    frontier_data = (target_returns, frontier_vols)
    
    return results_df, max_sharpe_portfolio, min_vol_portfolio, mean_returns, cov_matrix, frontier_data

if __name__ == "__main__":
    # Test block
    print("Optimizer loaded. Use run_monte_carlo(price_data) to execute.")
