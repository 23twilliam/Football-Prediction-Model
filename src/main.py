from pathlib import Path

from data import load_seasons
from backtest import dixon_coles_walk_forward_loop, xgb_walk_forward

# Starts at 13/14: football-data's shots-on-target definition changed in summer 2013 (SoT/shots
# ratio 0.55-0.57 before, 0.32-0.36 after, with total shots unchanged -- a measurement change,
# not a football one). Since FEATURE_COLS is SoT-based, pooling across that break would train on
# two different measurements. Pinnacle odds also only begin in 12/13.
SEASONS = [
    (f'E0{y % 100:02d}{(y + 1) % 100:02d}.csv', f'{y % 100:02d}/{(y + 1) % 100:02d}')
    for y in range(2013, 2025)
]

matches = load_seasons(Path('../data'), SEASONS)

results = xgb_walk_forward(matches)
for k, v in results.items():
    print(f"{k}: {v}")