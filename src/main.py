from pathlib import Path
from data import load_football_data_csv
from poisson_model import simple_match_probs, fit_attack_defence
import pandas as pd

matches = load_football_data_csv(Path('../data/E0.csv'), '24/25')

lam = matches['home_goals'].mean()
mu = matches['away_goals'].mean()

probs = simple_match_probs(lam, mu)
toy_matches = pd.DataFrame([
    {'home_team': 'A', 'away_team': 'B', 'home_goals': 3, 'away_goals': 1},
    {'home_team': 'A', 'away_team': 'C', 'home_goals': 2, 'away_goals': 0},
    {'home_team': 'A', 'away_team': 'D', 'home_goals': 4, 'away_goals': 0},
    {'home_team': 'B', 'away_team': 'C', 'home_goals': 1, 'away_goals': 1},
    {'home_team': 'B', 'away_team': 'D', 'home_goals': 2, 'away_goals': 1},
    {'home_team': 'C', 'away_team': 'D', 'home_goals': 1, 'away_goals': 0},
])
print(fit_attack_defence(toy_matches))