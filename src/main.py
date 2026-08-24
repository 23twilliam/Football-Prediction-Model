from pathlib import Path

from data import load_football_data_csv
from backtest import dixon_coles_walk_forward_loop, xgb_walk_forward

matches = load_football_data_csv(Path('../data/E0.csv'), '23/24')

results = xgb_walk_forward(matches)
for k, v in results.items():
    print(f"{k}: {v}")