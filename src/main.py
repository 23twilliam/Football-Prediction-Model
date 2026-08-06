from pathlib import Path

from data import load_football_data_csv
from backtest import walk_forward_loop

matches = load_football_data_csv(Path('../data/E0.csv'), '24/25')

results = walk_forward_loop(matches)
for k, v in results.items():
    print(f"{k}: {v}")