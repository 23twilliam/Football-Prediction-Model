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

ODDS_SOURCES = {
    'decision': ('decision_odds_home', 'decision_odds_draw', 'decision_odds_away'),   # Avg, ~4.5% vig
    'best':     ('best_odds_home', 'best_odds_draw', 'best_odds_away'),               # Max, ~1.0% vig
    'pinnacle': ('pinnacle_odds_home', 'pinnacle_odds_draw', 'pinnacle_odds_away'),   # ~3.6% vig
}


def odds_from(match, source: str) -> dict:
    home_col, draw_col, away_col = ODDS_SOURCES[source]
    return {'Home': match[home_col], 'Draw': match[draw_col], 'Away': match[away_col]}


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

def kelly_stake(model_prob, decimal_odds, fraction: float= 0.25, cap = 0.01):
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

def xgb_walk_forward_loop(matches_df: pd.DataFrame, training_days=730, retrain_every_n_days=10, min_edge=0.02,
                          balance=1000, settlement_odds='decision', n_models=10, feature_cols=None,
                          **model_kwargs):
    """settlement_odds selects the price bets are PAID at ('decision'/'best'/'pinnacle').
    The devig anchor stays on decision odds regardless, so predictions, log loss and bet
    selection are identical across settlement choices.

    training_days is a WARM-UP, not a rolling window: train_df is always every match before the
    current window, so this only controls when scoring starts, never how much data the model
    sees. 730 = two full seasons, chosen because venue-split 5-match rolling features need ~19
    home and ~19 away matches per team to fill, so one season leaves them still warming up and a
    promoted team with none. At the previous 60, the first two evaluated seasons carried 83% of
    all backtest losses purely from predicting on almost no history (FINDINGS.md §16)."""
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

        models = fit_ensemble(train_df, n_models=n_models, feature_cols=feature_cols, **model_kwargs)

        for idx, match in test_df.iterrows():
            row = test_df.loc[[idx]]
            probs, uncertainty = predict_match_ensemble(
                models, row[FEATURE_COLS if feature_cols is None else feature_cols], market_base_margin(row))

            raw_odds = odds_from(match, 'decision')        # probability anchor
            settle_odds = odds_from(match, settlement_odds)  # what the bet actually pays
            outcome = actual_outcome(match["home_goals"], match["away_goals"])
            market_probs = devig(raw_odds)

            best_outcome, best_edge = None, min_edge
            for result in ['Home', 'Draw', 'Away']:
                edge = probs[result] - market_probs[result]
                if edge > best_edge:
                    best_edge, best_outcome = edge, result

            stake = profit = 0.0
            bet_return = np.nan
            if best_outcome is not None:
                odds = settle_odds[best_outcome]
                stake = balance * kelly_stake(probs[best_outcome], odds)
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
                'odds_decision': raw_odds[best_outcome] if best_outcome else np.nan,
                'odds_best': odds_from(match, 'best')[best_outcome] if best_outcome else np.nan,
                'odds_pinnacle': odds_from(match, 'pinnacle')[best_outcome] if best_outcome else np.nan,
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


def xgb_walk_forward(matches_df: pd.DataFrame, training_days=730, retrain_every_n_days=6, min_edge=0.02,
                     balance=1000, settlement_odds='decision'):
    records = xgb_walk_forward_loop(
        matches_df, training_days=training_days, retrain_every_n_days=retrain_every_n_days,
        min_edge=min_edge, balance=balance, settlement_odds=settlement_odds,
    )
    return summarize_backtest(records, starting_balance=balance)
