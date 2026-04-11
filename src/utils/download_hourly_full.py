"""
Download full historical hourly OHLCV from Dukascopy (free, no signup).
Covers 2015-01-01 to 2026-04-10 for EUR/USD, GBP/JPY, XAU/USD.
"""
from datetime import datetime
import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_CROSSES_GBP_JPY,
    INSTRUMENT_FX_METALS_XAU_USD,
)
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "price", "hourly")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS = {
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
    "GBPJPY": INSTRUMENT_FX_CROSSES_GBP_JPY,
    "XAUUSD": INSTRUMENT_FX_METALS_XAU_USD,
}

START = datetime(2015, 1, 1)
END = datetime(2026, 4, 10)

for pair_name, instrument in PAIRS.items():
    print(f"\n{'='*60}")
    print(f"Downloading {pair_name} hourly from {START.date()} to {END.date()}...")
    print(f"{'='*60}")
    
    try:
        df = dukascopy_python.fetch(
            instrument=instrument,
            interval=dukascopy_python.INTERVAL_HOUR_1,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=START,
            end=END,
        )
        
        if df is None or df.empty:
            print(f"  WARNING: No data returned for {pair_name}")
            continue
        
        # Reset index (datetime index → column)
        df = df.reset_index()
        
        # Rename columns to match our spec
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'time' in cl or 'date' in cl or 'index' in cl:
                col_map[col] = 'datetime_utc'
            elif cl == 'open':
                col_map[col] = 'open'
            elif cl == 'high':
                col_map[col] = 'high'
            elif cl == 'low':
                col_map[col] = 'low'
            elif cl == 'close':
                col_map[col] = 'close'
            elif cl == 'volume':
                col_map[col] = 'volume'
        
        df = df.rename(columns=col_map)
        keep = [c for c in ['datetime_utc', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        df = df[keep]
        
        # Format datetime
        df['datetime_utc'] = df['datetime_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{pair_name}_hourly.csv")
        df.to_csv(out_path, index=False)
        
        print(f"  Rows: {len(df)}")
        print(f"  Range: {df['datetime_utc'].iloc[0]} to {df['datetime_utc'].iloc[-1]}")
        print(f"  Saved: {out_path}")
        
    except Exception as e:
        print(f"  ERROR for {pair_name}: {e}")

print("\nDone.")
