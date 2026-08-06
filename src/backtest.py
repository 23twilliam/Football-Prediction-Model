import numpy as np
import pandas as pd

from poisson_model import fit_attack_defence, simple_match_probs

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
    training_days = 30
    retrain_every_n_days = 15

    min_edge = 0.02
    balance = 1000

    matches_df = matches_df.sort_values(by=['date'])
    window_start = matches_df.date.min() + pd.Timedelta(days=training_days)
    last_date = matches_df.date.max()

    log_loss_ = []
    market_baseline = []

    bet_profits = []
    bet_returns = []
    stakes = []

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
            teams, attack, defence, home_adv, rho = fit_attack_defence(train_df)
            for _, match in test_df.iterrows():
                home_idx = teams.index(match["home_team"])
                away_idx = teams.index(match['away_team'])
                lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
                mu = np.exp(attack[away_idx] - defence[home_idx])

                probs = simple_match_probs(lam=lam, mu=mu, rho=rho)

                raw_odds = {
                    'Home': match['decision_odds_home'],
                    'Draw': match['decision_odds_draw'],
                    'Away': match['decision_odds_away'],
                }
                outcome = actual_outcome(match["home_goals"], match["away_goals"])

                market_probs = devig(raw_odds)

                log_loss_.append(log_loss(actual_outcome=outcome, probs=probs))
                market_baseline.append(log_loss(actual_outcome=outcome, probs=market_probs))

                best_outcome = None
                best_edge = min_edge

                for result in ['Home', 'Draw', 'Away']:
                    edge = probs[result] - market_probs[result]

                    if edge > best_edge:
                        best_edge = edge
                        best_outcome = result

                if best_outcome is not None:
                    odds = raw_odds[best_outcome]

                    stake_fraction = kelly_stake(probs[best_outcome], odds)
                    stake = balance * stake_fraction

                    if outcome == best_outcome:
                        profit = stake * (odds - 1)
                    else:
                        profit = -stake

                    balance += profit

                    bet_profits.append(profit)
                    bet_returns.append(profit / stake if stake > 0 else 0)
                    stakes.append(stake)

            window_start += pd.Timedelta(days=retrain_every_n_days)

    return {
        'avg_model_log_loss': np.mean(log_loss_),
        'avg_market_log_loss': np.mean(market_baseline),
        'n_bets': len(bet_profits),
        'total_profit': sum(bet_profits),
        'roi': sum(bet_profits) / sum(stakes) if stakes else None,
        'max_drawdown': max_drawdown(bet_profits) if bet_profits else None,
        'sharpe_ratio': sharpe_ratio(bet_returns) if len(bet_returns) > 1 else None,
        'final_balance': balance,
    }
