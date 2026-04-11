"""
Resample hourly OHLCV to 4-hour bars.
Input:  data/raw/price/hourly/{PAIR}_hourly.csv
Output: data/raw/price/{PAIR}_4h.csv

6 bars per trading day, each naturally aligned with a market session:
  00:00-04:00  Late Sydney / Early Asian
  04:00-08:00  Asian / Early London
  08:00-12:00  London session
  12:00-16:00  London-NY overlap (peak liquidity)
  16:00-20:00  NY session
  20:00-00:00  Late NY / Early Sydney
"""
import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "price")
HOURLY_DIR = os.path.join(BASE, "hourly")

PAIRS = ["EURUSD", "GBPJPY", "XAUUSD"]

for pair in PAIRS:
    hourly_path = os.path.join(HOURLY_DIR, f"{pair}_hourly.csv")
    
    if not os.path.exists(hourly_path):
        print(f"  SKIP: {hourly_path} not found")
        continue
    
    print(f"\nResampling {pair} hourly → 4H...")
    
    df = pd.read_csv(hourly_path, parse_dates=["datetime_utc"])
    df = df.set_index("datetime_utc")
    df = df.sort_index()
    
    # Resample to 4H bars
    resampled = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum" if "volume" in df.columns else "first",
    }).dropna(subset=["open"])  # drop bars with no data (weekends)
    
    # Reset index
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={"datetime_utc": "datetime_utc"})
    
    # Add session label for each 4H bar
    hour = resampled["datetime_utc"].dt.hour
    session_map = {
        0: "late_sydney_early_asian",
        4: "asian_early_london",
        8: "london",
        12: "london_ny_overlap",
        16: "ny",
        20: "late_ny_early_sydney",
    }
    resampled["session"] = hour.map(session_map)
    
    # Add date column (for merging with daily macro/sentiment data)
    resampled["date"] = resampled["datetime_utc"].dt.date.astype(str)
    
    # Save
    out_path = os.path.join(BASE, f"{pair}_4h.csv")
    resampled.to_csv(out_path, index=False)
    
    print(f"  Rows: {len(resampled)} (from {len(df)} hourly bars)")
    print(f"  Range: {resampled['datetime_utc'].iloc[0]} to {resampled['datetime_utc'].iloc[-1]}")
    print(f"  Sessions per day: {resampled.groupby('date').size().median():.0f}")
    print(f"  Saved: {out_path}")

print("\nDone. 4H bars saved to data/raw/price/")
