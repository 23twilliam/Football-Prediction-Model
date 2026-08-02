from pathlib import Path
import numpy as np
from data import load_football_data_csv
from poisson_model import simple_match_probs, fit_attack_defence, tau_correction

matches = load_football_data_csv(Path('../data/E0.csv'), '24/25')

teams, attack, defense, home_adv, rho = fit_attack_defence(matches)
i = teams.index("Nott'm Forest")
j = teams.index("Man United")
lam = np.exp(attack[i] - defense[j] + home_adv)
mu = np.exp(attack[j] - defense[i])

probs = simple_match_probs(lam, mu)
print(probs)