import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_efficient_frontier(results_df, max_sharpe_portfolio, min_vol_portfolio, frontier_data=None, risk_free_rate=0.0428, output_dir="results", filename="efficient_frontier.png"):
    """
    Visualizes the Monte Carlo simulations and highlights optimal portfolios.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing Monte Carlo simulation results.
        max_sharpe_portfolio (pd.Series): Data for the maximum Sharpe ratio portfolio.
        min_vol_portfolio (pd.Series): Data for the minimum volatility portfolio.
        output_dir (str): Directory to save the plot.
        filename (str): Name of the generated plot file.
    """
    print("Generating Efficient Frontier visualization...")
    
    # Set plot style for a more professional resume-ready look
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 7))
    
    # Scatter plot of all simulated portfolios, colored by Sharpe Ratio
    scatter = plt.scatter(
        results_df['Volatility'], 
        results_df['Return'], 
        c=results_df['Sharpe Ratio'], 
        cmap='viridis', 
        marker='o', 
        s=10, 
        alpha=0.6
    )
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Highlight Max Sharpe Portfolio
    plt.scatter(
        max_sharpe_portfolio['Volatility'], 
        max_sharpe_portfolio['Return'], 
        marker='*', 
        color='gold', 
        s=400, 
        edgecolor='black',
        label='Max Sharpe Portfolio'
    )
    
    # Highlight Minimum Volatility Portfolio
    plt.scatter(
        min_vol_portfolio['Volatility'], 
        min_vol_portfolio['Return'], 
        marker='*', 
        color='red', 
        s=400, 
        edgecolor='black',
        label='Min Volatility Portfolio'
    )
    
    # Labels and Titles
    plt.title('Portfolio Optimization: Efficient Frontier\n(META, AMZN, MELI, NU)', fontsize=16, fontweight='bold')
    plt.xlabel('Volatility (Expected Risk)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    
    # Extract weights for the text box
    def format_weights(portfolio, title):
        lines = [f"{title}:"]
        for idx in portfolio.index:
            if "Weight" in idx:
                ticker = idx.replace(' Weight', '')
                lines.append(f"  {ticker}: {portfolio[idx]*100:.1f}%")
        return '\n'.join(lines)
        
    text_content = f"{format_weights(max_sharpe_portfolio, 'Max Sharpe')}\n\n{format_weights(min_vol_portfolio, 'Min Vol')}"
    
    # Add a text box to the bottom right
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.text(0.95, 0.05, text_content, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)
             
    if frontier_data:
        target_returns, frontier_vols = frontier_data
        plt.plot(frontier_vols, target_returns, 'k--', linewidth=2, label='Continuous Frontier')
        
    # Plot Capital Allocation Line (CAL)
    max_vol = results_df['Volatility'].max()
    cal_x = [0, max_vol]
    cal_y = [risk_free_rate, risk_free_rate + max_sharpe_portfolio['Sharpe Ratio'] * max_vol]
    plt.plot(cal_x, cal_y, 'b:', linewidth=2, label='Capital Allocation Line (CAL)')
    
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.xlim(left=0) # ensure y-axis intercept is visible
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # Ensure plot is closed to free up resources
    plt.close()

if __name__ == "__main__":
    print("Visualizer loaded. Use plot_efficient_frontier() to generate graphs.")
