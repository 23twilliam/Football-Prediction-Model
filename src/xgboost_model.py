import xgboost as xg
import pandas as pd
import numpy as np

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

