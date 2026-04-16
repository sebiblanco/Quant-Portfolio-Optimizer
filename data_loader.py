import yfinance as yf
import pandas as pd
import os

def download_data(tickers, period="5y", output_dir="data", filename="historical_prices.csv"):
    """
    Downloads historical adjusted close prices for a list of tickers.
    
    Args:
        tickers (list): List of ticker symbols as strings.
        period (str): The time period to download (default is 5 years).
        output_dir (str): Directory to save the csv file.
        filename (str): Name of the csv file.
        
    Returns:
        pd.DataFrame: DataFrame containing adjusted close prices.
    """
    print(f"Downloading {period} data for {tickers}...")
    
    # Download data
    data = yf.download(tickers, period=period)
    
    # Check if 'Adj Close' is available, else fallback to 'Close'
    if 'Adj Close' in data.columns:
        price_data = data['Adj Close']
    elif 'Close' in data.columns:
        price_data = data['Close']
        print("Warning: 'Adj Close' not found. Using 'Close' prices instead.")
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' were found in the downloaded data.")
        
    # Drop rows with NaN values to ensure clean data for calculations
    price_data = price_data.dropna()
    print(f"Data successfully fetched. Shape: {price_data.shape}")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    price_data.to_csv(output_path)
    print(f"Data saved to {output_path}")
    
    return price_data

if __name__ == "__main__":
    download_data(["META", "AMZN", "MELI", "NU"])
