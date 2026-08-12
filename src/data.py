from pathlib import Path
import pandas as pd

SCHEMA_COLUMN_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",

    "AvgH": "decision_odds_home",
    "AvgD": "decision_odds_draw",
    "AvgA": "decision_odds_away",

    "AvgCH": "closing_odds_home",
    "AvgCD": "closing_odds_draw",
    "AvgCA": "closing_odds_away",
}


def load_football_data_csv(path: Path, season: str):
    raw = pd.read_csv(path)
    raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=True)
    raw = raw[list(SCHEMA_COLUMN_MAP.keys())]
    raw["season"] = season
    renamed = raw.rename(columns=SCHEMA_COLUMN_MAP)
    return renamed


def build_team_events(matches_df: pd.DataFrame):
    home_events = matches_df[['date', 'home_team', 'home_goals', 'away_goals']].rename(
        columns={'home_team': 'team', 'home_goals': 'goals_for', 'away_goals': 'goals_against'}
    )
    away_events = matches_df[['date', 'away_team', 'away_goals', 'home_goals']].rename(
        columns={'away_team': 'team', 'away_goals': 'goals_for', 'home_goals': 'goals_against'}
    )

    team_events = pd.concat([home_events, away_events])
    team_events = team_events.sort_values(by=['team', 'date']).reset_index(drop=True)

    # Added features of team events
    team_events['rolling_goals_for'] = team_events.groupby('team')['goals_for'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    team_events['rolling_goals_against'] = team_events.groupby('team')['goals_against'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    team_events['days_since_last'] = team_events.groupby('team')['date'].diff().dt.days

    team_events['matches_last_14d'] = team_events.groupby('team').apply(
        lambda x: x.set_index('date')['team'].rolling('14D', closed='left').count()
    ).reset_index(level=0, drop=True).fillna(0)

    return team_events


if __name__ == '__main__':
    dataframe = load_football_data_csv(Path('../data/E0.csv'), '24/25')
    print(dataframe.head())
