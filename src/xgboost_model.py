import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from pathlib import Path

from data import load_football_data_csv

def prepare_data(matches_df: pd.DataFrame):
    # 0 home
    # 1 draw
    # 2 away
    matches_df["result"] = np.select(
        [matches_df["home_goals"] > matches_df["away_goals"],
         matches_df["home_goals"] == matches_df["away_goals"],
         matches_df["home_goals"] < matches_df["away_goals"]],
        [0,1,2],
        default = 1
    )

    teams = sorted(pd.concat([matches_df["home_team"], matches_df["away_team"]]).unique())
    team_to_idx = {team: i for i, team in enumerate(teams)}

    matches_df['home_team'] = matches_df['home_team'].map(team_to_idx)
    matches_df['away_team'] = matches_df['away_team'].map(team_to_idx)

    return matches_df, teams, team_to_idx

def fit_classifier(
        matches_df: pd.DataFrame,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
):
    matches_df, teams, team_to_idx = prepare_data(matches_df)
    features = ["home_team", "away_team"]

    x = matches_df[features].copy()
    y = matches_df["result"]

    model = XGBClassifier(
        objective = "multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,

        eval_metric = "mlogloss",
        random_state = 42
    )

    model.fit(x, y)

    return model, teams, team_to_idx

def predict_match(
    model,
    team_to_idx,
    home_team,
    away_team
):
    X = pd.DataFrame({
        "home_team": [team_to_idx[home_team]],
        "away_team": [team_to_idx[away_team]]
    })

    probabilities = model.predict_proba(X)[0]

    return {
        "Home": probabilities[2],
        "Draw": probabilities[1],
        "Away": probabilities[0]
    }

if __name__ == '__main__':
    matches_df = load_football_data_csv(Path('../data/E0.csv'), '24/25')
    model, teams, team_to_idx = fit_classifier(matches_df)

    probs = predict_match(
        model,
        team_to_idx,
        "Liverpool",
        "Arsenal"
    )

    print(probs)