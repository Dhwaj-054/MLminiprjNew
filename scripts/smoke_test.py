"""Simple smoke test to verify data can be loaded and module imports."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data_moods.csv'

def main():
    if not DATA_PATH.exists():
        print(f"ERROR: data file not found at {DATA_PATH}")
        sys.exit(2)
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    print('Data loaded successfully; shape=', df.shape)
    print('Columns:', df.columns.tolist())

if __name__ == '__main__':
    main()
