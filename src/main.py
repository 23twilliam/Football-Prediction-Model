from pathlib import Path

from data import load_seasons
from backtest import dixon_coles_walk_forward_loop, xgb_walk_forward

SEASONS = [
    ('E01920.csv', '19/20'),
    ('E02021.csv', '20/21'),
    ('E02122.csv', '21/22'),
    ('E02223.csv', '22/23'),
    ('E02324.csv', '23/24'),
    ('E02425.csv', '24/25'),
]

matches = load_seasons(Path('../data'), SEASONS)

results = xgb_walk_forward(matches)
for k, v in results.items():
    print(f"{k}: {v}")