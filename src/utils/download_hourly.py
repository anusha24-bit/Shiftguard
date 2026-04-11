"""
Download hourly OHLCV for EUR/USD, GBP/JPY, XAU/USD via yfinance.
yfinance provides ~730 days of hourly data.
Output: data/raw/price/hourly/{PAIR}_hourly.csv
"""
import yfinance as yf
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "price", "hourly")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPJPY": "GBPJPY=X",
    "XAUUSD": "GC=F",
}

for pair_name, ticker in PAIRS.items():
    print(f"\nDownloading {pair_name} ({ticker}) hourly...")
    
    df = yf.download(ticker, interval="1h", period="730d", progress=True)
    
    if df.empty:
        print(f"  WARNING: No data returned for {pair_name}")
        continue
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to get datetime as column
    df = df.reset_index()
    
    # Rename columns to match our spec
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "datetime" in col_lower or "date" in col_lower:
            rename_map[col] = "datetime_utc"
        elif col_lower == "open":
            rename_map[col] = "open"
        elif col_lower == "high":
            rename_map[col] = "high"
        elif col_lower == "low":
            rename_map[col] = "low"
        elif col_lower == "close":
            rename_map[col] = "close"
        elif col_lower == "volume":
            rename_map[col] = "volume"
    
    df = df.rename(columns=rename_map)
    
    # Keep only the columns we need
    keep_cols = [c for c in ["datetime_utc", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols]
    
    # Convert datetime to UTC string
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, f"{pair_name}_hourly.csv")
    df.to_csv(out_path, index=False)
    
    date_range = f"{df['datetime_utc'].iloc[0]} to {df['datetime_utc'].iloc[-1]}"
    print(f"  Saved: {out_path}")
    print(f"  Rows: {len(df)} | Range: {date_range}")

print("\nDone. Hourly data saved to data/raw/price/hourly/")
