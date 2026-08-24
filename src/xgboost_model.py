import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from pathlib import Path

from data import load_football_data_csv, build_team_events, merge_team_events

FEATURE_COLS = [
    'home_rolling_mean_goals_for', 'home_rolling_mean_goals_against',
    'home_days_since_last', 'home_matches_last_14d',
    'away_rolling_mean_goals_for', 'away_rolling_mean_goals_against',
    'away_days_since_last', 'away_matches_last_14d',
]

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
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 30,
        reg_lambda: float = 5,
):
    """matches_df must already be prepared via prepare_data()."""
    x = matches_df[FEATURE_COLS]
    y = matches_df["result"]
    base_margin = market_base_margin(matches_df)

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

        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(x, y, base_margin=base_margin)

    return model


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


if __name__ == '__main__':
    raw_matches = load_football_data_csv(Path('../data/E0.csv'), '24/25')
    matches_df = prepare_data(raw_matches)

    model = fit_classifier(matches_df)

    for idx in matches_df.index:
        row = matches_df.loc[[idx]]
        probs = predict_match(model, row[FEATURE_COLS], market_base_margin(row))

        print(probs)
