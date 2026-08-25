import numpy as np
import pandas as pd

from poisson_model import fit_attack_defence, simple_match_probs
from xgboost_model import prepare_data, fit_classifier, predict_match, FEATURE_COLS, market_base_margin, \
    fit_ensemble, predict_match_ensemble

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

def conservative_prob(mean_prob: float, std: float, certainty_cap: float = 0.75) -> float:
    """Risk-adjusted probability used for betting decisions
    1. Shrink by one ensemble standard deviation distrust bets where the models
       disagree with each other
    2. A hard ceiling regardless of agreement
    """
    return min(max(mean_prob - std, 0.0), certainty_cap)


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

def dixon_coles_walk_forward_loop(matches_df: pd.DataFrame, training_days = 30, retrain_every_n_days = 15,
                                  min_edge = 0.02, balance = 1000):

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

def xgb_walk_forward_loop(matches_df: pd.DataFrame, training_days=60, retrain_every_n_days=10, min_edge=0.02,
                          max_edge=0.06, balance=1000):
    matches_df = prepare_data(matches_df)   # ONCE, globally, before any slicing
    matches_df = matches_df.sort_values(by=['date'])
    window_start = matches_df.date.min() + pd.Timedelta(days=training_days)
    last_date = matches_df.date.max()

    records = []

    while window_start < last_date:
        train_df = matches_df[matches_df['date'] < window_start].copy()
        test_df = matches_df[
            (matches_df['date'] >= window_start) &
            (matches_df['date'] < window_start + pd.Timedelta(days=retrain_every_n_days))
        ]

        if len(train_df) < training_days or test_df.empty:
            window_start += pd.Timedelta(days=retrain_every_n_days)
            continue

        models = fit_ensemble(train_df)

        for idx, match in test_df.iterrows():
            row = test_df.loc[[idx]]
            probs, uncertainty = predict_match_ensemble(models, row[FEATURE_COLS], market_base_margin(row))

            raw_odds = {'Home': match['decision_odds_home'], 'Draw': match['decision_odds_draw'], 'Away': match['decision_odds_away']}
            outcome = actual_outcome(match["home_goals"], match["away_goals"])
            market_probs = devig(raw_odds)

            # Bet selection and sizing both act on the risk-adjusted probability
            # calibration signal, unaffected by staking policy.
            conservative_probs = {r: conservative_prob(probs[r], uncertainty[r]) for r in ['Home', 'Draw', 'Away']}

            # Edges beyond max_edge are excluded: since the three edges
            # sum to zero (model and market probabilities each sum to 1), an implausibly large
            # edge is the only candidate most matches ever produce, so this mostly means
            # skipping the match
            best_outcome, best_edge = None, min_edge
            for result in ['Home', 'Draw', 'Away']:
                edge = conservative_probs[result] - market_probs[result]
                if min_edge < edge <= max_edge and edge > best_edge:
                    best_edge, best_outcome = edge, result

            stake = profit = 0.0
            bet_return = np.nan
            if best_outcome is not None:
                odds = raw_odds[best_outcome]
                stake = balance * kelly_stake(conservative_probs[best_outcome], odds)
                profit = stake * (odds - 1) if outcome == best_outcome else -stake
                balance += profit
                bet_return = profit / stake if stake > 0 else 0.0

            records.append({
                'date': match['date'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'home_goals': match['home_goals'],
                'away_goals': match['away_goals'],
                'margin': abs(match['home_goals'] - match['away_goals']),
                'actual_outcome': outcome,
                'prob_home': probs['Home'], 'prob_draw': probs['Draw'], 'prob_away': probs['Away'],
                'prob_home_std': uncertainty['Home'], 'prob_draw_std': uncertainty['Draw'], 'prob_away_std': uncertainty['Away'],
                'market_home': market_probs['Home'], 'market_draw': market_probs['Draw'], 'market_away': market_probs['Away'],
                'model_log_loss': log_loss(actual_outcome=outcome, probs=probs),
                'market_log_loss': log_loss(actual_outcome=outcome, probs=market_probs),
                'best_outcome': best_outcome,
                'best_edge': best_edge if best_outcome is not None else np.nan,
                'bet_placed': best_outcome is not None,
                'stake': stake,
                'profit': profit,
                'bet_return': bet_return,
            })

        window_start += pd.Timedelta(days=retrain_every_n_days)

    return pd.DataFrame.from_records(records)


def summarize_backtest(records: pd.DataFrame, starting_balance=1000) -> dict:
    bets = records[records['bet_placed']]
    bet_profits = bets['profit'].tolist()
    bet_returns = bets['bet_return'].tolist()

    return {
        'avg_model_log_loss': records['model_log_loss'].mean(),
        'avg_market_log_loss': records['market_log_loss'].mean(),
        'n_bets': len(bets),
        'total_profit': bets['profit'].sum(),
        'roi': bets['profit'].sum() / bets['stake'].sum() if len(bets) else None,
        'max_drawdown': max_drawdown(bet_profits) if bet_profits else None,
        'sharpe_ratio': sharpe_ratio(bet_returns) if len(bet_returns) > 1 else None,
        'final_balance': starting_balance + bets['profit'].sum(),
    }


def xgb_walk_forward(matches_df: pd.DataFrame, training_days=60, retrain_every_n_days=6, min_edge=0.02, max_edge=0.06,
                     balance=1000):
    records = xgb_walk_forward_loop(
        matches_df, training_days=training_days, retrain_every_n_days=retrain_every_n_days,
        min_edge=min_edge, max_edge=max_edge, balance=balance,
    )
    return summarize_backtest(records, starting_balance=balance)
