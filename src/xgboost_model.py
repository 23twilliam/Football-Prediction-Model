import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from pathlib import Path

from data import load_football_data_csv, build_team_events, merge_team_events


def prepare_data(matches: pd.DataFrame):
    # 0 home
    # 1 draw
    # 2 away

    matches = matches.copy()

    team_events = build_team_events(matches)
    matches_df = merge_team_events(matches, team_events)

    matches_df["result"] = np.select(
        [matches_df["home_goals"] > matches_df["away_goals"],
         matches_df["home_goals"] == matches_df["away_goals"],
         matches_df["home_goals"] < matches_df["away_goals"]],
        [0, 1, 2],
        default=1
    )

    return matches_df


def fit_classifier(
        matches_df: pd.DataFrame,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
):
    matches_df = prepare_data(matches_df)
    features = [
        'home_rolling_mean_goals_for', 'home_rolling_mean_goals_against',
        'home_days_since_last', 'home_matches_last_14d',
        'away_rolling_mean_goals_for', 'away_rolling_mean_goals_against',
        'away_days_since_last', 'away_matches_last_14d',
    ]

    x = matches_df[features].copy()
    y = matches_df["result"]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,

        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(x, y)

    return model


def predict_match(
        model,
        x_row
):
    probabilities = model.predict_proba(x_row)[0]

    return {
        "Home": probabilities[0],
        "Draw": probabilities[1],
        "Away": probabilities[2]
    }


if __name__ == '__main__':
    matches_df = load_football_data_csv(Path('../data/E0.csv'), '24/25')
    team_events = build_team_events(matches_df)
    print("team_events columns:", team_events.columns.tolist())

    merged = merge_team_events(matches_df, team_events)
    print("merged columns:", merged.columns.tolist())

    model = fit_classifier(matches_df)
    feature_cols = [
        'home_rolling_mean_goals_for', 'home_rolling_mean_goals_against',
        'home_days_since_last', 'home_matches_last_14d',
        'away_rolling_mean_goals_for', 'away_rolling_mean_goals_against',
        'away_days_since_last', 'away_matches_last_14d',
    ]

    for idx in merged.index:
        x_row = merged.loc[[idx], feature_cols]
        probs = predict_match(model, x_row)

        print(probs)
