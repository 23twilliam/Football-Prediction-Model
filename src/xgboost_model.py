import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from pathlib import Path

from data import load_football_data_csv, build_team_events, merge_team_events

RECENCY_COLS = [
    'home_days_since_last', 'home_matches_last_14d',
    'away_days_since_last', 'away_matches_last_14d',
]


def feature_set(metrics) -> list:
    """Venue-split rolling form features for the given metrics, plus recency. Metrics must be in
    data.ROLLING_METRICS. They are correlated (goals<->SoT 0.71, shots<->SoT 0.81), i.e. one
    signal at different noise levels rather than several, so combinations underperform the best
    single metric."""
    return [f'{side}_rolling_mean_{m}_{d}'
            for side in ('home', 'away') for m in metrics for d in ('for', 'against')] + RECENCY_COLS


# Shots on target ALONE beat every combination tested, including goals-only and all three
# Goals are the noisiest measure of attacking strength, so once a
# cleaner measure of the same quantity is present they add variance, not information.
FEATURE_COLS = feature_set(['shots_on_target'])

ODDS_COLS = ['decision_odds_home', 'decision_odds_draw', 'decision_odds_away']  # class order 0=Home,1=Draw,2=Away


def market_base_margin(matches_df: pd.DataFrame) -> np.ndarray:
    """Raw per-class log-odds implied by the bookmaker's odds (log(1/odds), un-devigged).
    Passed to XGBoost as base_margin: softmax renormalises automatically, so softmax of this
    reproduces the devigged market probabilities exactly as the model's starting point:
    trees then only need to fit the residual against the market."""
    return -np.log(matches_df[ODDS_COLS].to_numpy())


def prepare_data(matches: pd.DataFrame) -> pd.DataFrame:
    # 0 home
    # 1 draw
    # 2 away

    matches = matches.copy()

    team_events = build_team_events(matches)
    matches = merge_team_events(matches, team_events)

    matches["result"] = np.select(
        [matches["home_goals"] > matches["away_goals"],
         matches["home_goals"] == matches["away_goals"],
         matches["home_goals"] < matches["away_goals"]],
        [0, 1, 2],
        default=1
    )

    return matches


def fit_classifier(
        matches_df: pd.DataFrame,
        n_estimators: int = 150,
        max_depth: int = 1,
        learning_rate: float = 0.01,
        subsample: float = 1.0,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 30,
        reg_lambda: float = 5,
        reg_alpha: float = 0,
        gamma: float = 0,
        random_state: int = 42,
        feature_cols=None,
        time_decay: float = 0.0,
):
    """matches_df must already be prepared via prepare_data().

    time_decay: exponential recency weighting, exp(-time_decay * days_before_latest_training
    match). 0 disables it (every match counts equally, however old). Same form as
    poisson_model.fit_attack_defence, which uses 0.002. Useful half-lives: 0.0005 ~ 3.8y,
    0.001 ~ 1.9y, 0.002 ~ 0.95y, 0.004 ~ 0.5y. Matters more now that walk-forward windows span
    up to 11 years, over which overround fell 6.1% -> 4.1% and home-win rate ranged 38-49%."""
    x = matches_df[FEATURE_COLS if feature_cols is None else feature_cols]
    y = matches_df["result"]
    base_margin = market_base_margin(matches_df)

    sample_weight = None
    if time_decay > 0:
        days_ago = (matches_df['date'].max() - matches_df['date']).dt.days
        sample_weight = np.exp(-time_decay * days_ago)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        gamma=gamma,

        eval_metric="mlogloss",
        random_state=random_state
    )

    model.fit(x, y, base_margin=base_margin, sample_weight=sample_weight)

    return model


def fit_ensemble(matches_df: pd.DataFrame, n_models: int = 10, seed: int = 42, **kwargs):
    """Bags n_models XGBoost fits, each with a different random_state (varies row/column
    subsampling). Disagreement across the ensemble's predictions on the same match is a
    proxy for how sensitive this walk-forward window's small training set (as few as 60-300
    rows) is to which rows happened to get sampled"""
    return [fit_classifier(matches_df, random_state=seed + i, **kwargs) for i in range(n_models)]


def predict_match(
        model,
        x_row,
        base_margin
):
    probabilities = model.predict_proba(x_row, base_margin=base_margin)[0]

    return {
        "Home": probabilities[0],
        "Draw": probabilities[1],
        "Away": probabilities[2]
    }


def predict_match_ensemble(models, x_row, base_margin):
    """Mean and std of the bagged ensemble's predicted probabilities for one match. High std on
    an outcome means this window's training data could have produced a very different
    prediction for it less trusting an edge computed from it."""
    all_probs = np.array([m.predict_proba(x_row, base_margin=base_margin)[0] for m in models])
    mean = all_probs.mean(axis=0)
    std = all_probs.std(axis=0)

    probs = {"Home": mean[0], "Draw": mean[1], "Away": mean[2]}
    uncertainty = {"Home": std[0], "Draw": std[1], "Away": std[2]}
    return probs, uncertainty


if __name__ == '__main__':
    raw_matches = load_football_data_csv(Path('../data/E0.csv'), '24/25')
    matches_df = prepare_data(raw_matches)

    model = fit_classifier(matches_df)

    for idx in matches_df.index:
        row = matches_df.loc[[idx]]
        probs = predict_match(model, row[FEATURE_COLS], market_base_margin(row))

        print(probs)
