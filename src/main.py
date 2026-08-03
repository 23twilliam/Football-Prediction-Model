from pathlib import Path
import numpy as np
from data import load_football_data_csv
from poisson_model import simple_match_probs, fit_attack_defence
import pandas as pd
from staking import kelly_stake

print(kelly_stake(model_prob=0.8, decimal_odds=3.00))

matches = load_football_data_csv(Path('../data/E0.csv'), '24/25')

tiny_matches = pd.DataFrame([
    {'date': pd.Timestamp('2024-08-01'), 'home_team': 'A', 'away_team': 'B', 'home_goals': 3, 'away_goals': 0},
    {'date': pd.Timestamp('2024-08-08'), 'home_team': 'B', 'away_team': 'A', 'home_goals': 0, 'away_goals': 3},
    {'date': pd.Timestamp('2024-08-15'), 'home_team': 'A', 'away_team': 'B', 'home_goals': 4, 'away_goals': 0},
])

teams, attack, defence, home_adv, rho = fit_attack_defence(tiny_matches)
print(dict(zip(teams, attack)))
print(dict(zip(teams, defence)))

i = teams.index("A")
j = teams.index("B")
lam = np.exp(attack[i] - defence[j] + home_adv)
mu = np.exp(attack[j] - defence[i])

probs = simple_match_probs(lam, mu)
print(lam, mu)
print(probs)