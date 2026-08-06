import numpy as np
import pandas as pd

from src.poisson_model import fit_attack_defence, simple_match_probs

def actual_outcome(home_goals, away_goals):
    if home_goals > away_goals:
        return 'Home'
    elif home_goals == away_goals:
        return 'Draw'
    else:
        return 'Away'

def devig(raw_odds):
    # Convert each to implied probabilit
    home = 1 / raw_odds["Home"]
    draw = 1 / raw_odds["Draw"]
    away = 1 / raw_odds["Away"]

    # Find overreach (odds_value) should be over 1
    odds_value = home + draw + away

    # Normalise to equal 1 when added
    home = home / odds_value
    draw = draw / odds_value
    away = away / odds_value

    return {'Home': home, 'Draw': draw, 'Away': away}

def kelly_stake(model_prob, decimal_odds, fraction: float= 0.25, cap = 0.025):
    b = decimal_odds - 1 # Net odds, profit per unit staked if win
    edge = model_prob * decimal_odds - 1
    if edge <= 0:
        return 0.0
    else:
        full_kelly = edge / b
        conservative_kelly = full_kelly * fraction # Quarter by default
        return min(conservative_kelly, cap)

def max_drawdown(profits):
    cum_profit = np.concatenate([[0], (np.cumsum(profits))])
    running_peek = np.maximum.accumulate(cum_profit)
    drawdown = running_peek - cum_profit
    return max(drawdown)

def sharpe_ratio(returns):
    # Take returns (per bet profit, as fraction of stake)
    mean_return = np.mean(returns)
    std_dev = np.std(returns, ddof=1) # treating as sample rather than entire pop of all possible bets
    return mean_return / std_dev

def log_loss(actual_outcome, probs):
    return -np.log(probs[actual_outcome])

def walk_forward_loop(matches_df):
    training_days = 3
    retrain_every_n_days = 2
    matches_df = matches_df.sort_values(by=['date'])
    window_start = matches_df.date.min() + pd.Timedelta(days=training_days)
    last_date = matches_df.date.max()

    while window_start < last_date:
        train_df = matches_df[matches_df['date'] < window_start].copy()
        test_df = matches_df[
            (matches_df['date'] >= window_start) &
            (matches_df['date'] < window_start + pd.Timedelta(days=retrain_every_n_days))
        ]
        if len(train_df) < training_days or test_df.empty:
            window_start += pd.Timedelta(days=retrain_every_n_days)
            continue
        else:
            log_loss_ = []
            market_baseline = []
            teams, attack, defence, home_adv, rho = fit_attack_defence(train_df)
            for _, match in test_df.iterrows():
                home_idx = teams.index(match["home_team"])
                away_idx = teams.index(match['away_team'])
                lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
                mu = np.exp(attack[away_idx] - defence[home_idx])

                probs = simple_match_probs(lam=lam, mu=mu, rho=rho)
                outcome = actual_outcome(match["home_goals"], match["away_goals"])
                log_loss_.append(log_loss(actual_outcome=outcome, probs=probs))
                market_baseline.append(log_loss(actual_outcome=outcome, probs=probs)) # TODO: Fix


            window_start += pd.Timedelta(days=retrain_every_n_days)


if __name__ == '__main__':
    tiny_matches = pd.DataFrame([
        {'date': pd.Timestamp('2024-08-01'), 'home_team': 'A', 'away_team': 'B', 'home_goals': 3, 'away_goals': 0},
        {'date': pd.Timestamp('2024-08-08'), 'home_team': 'B', 'away_team': 'A', 'home_goals': 0, 'away_goals': 3},
        {'date': pd.Timestamp('2024-08-15'), 'home_team': 'A', 'away_team': 'B', 'home_goals': 4, 'away_goals': 0},
        {'date': pd.Timestamp('2024-08-01'), 'home_team': 'A', 'away_team': 'B', 'home_goals': 3, 'away_goals': 0},
        {'date': pd.Timestamp('2024-08-08'), 'home_team': 'B', 'away_team': 'A', 'home_goals': 0, 'away_goals': 3},
        {'date': pd.Timestamp('2024-08-15'), 'home_team': 'A', 'away_team': 'B', 'home_goals': 4, 'away_goals': 0},
        {'date': pd.Timestamp('2024-08-01'), 'home_team': 'A', 'away_team': 'B', 'home_goals': 3, 'away_goals': 0},
        {'date': pd.Timestamp('2024-08-08'), 'home_team': 'B', 'away_team': 'A', 'home_goals': 0, 'away_goals': 3},
        {'date': pd.Timestamp('2024-08-15'), 'home_team': 'A', 'away_team': 'B', 'home_goals': 4, 'away_goals': 0},
    ])
    walk_forward_loop(tiny_matches)
