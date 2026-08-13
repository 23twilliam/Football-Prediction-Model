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

def merge_team_events(matches_df: pd.DataFrame, team_events: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ['team', 'date', 'rolling_mean_goals_for', 'rolling_mean_goals_against',
                    'days_since_last', 'matches_last_14d']

    home_events = team_events[feature_cols].rename(columns={
        'team': 'home_team',
        'rolling_mean_goals_for': 'home_rolling_mean_goals_for',
        'rolling_mean_goals_against': 'home_rolling_mean_goals_against',
        'days_since_last': 'home_days_since_last',
        'matches_last_14d': 'home_matches_last_14d',
    })
    matches_df = matches_df.merge(home_events, how='left', on=['home_team', 'date'])

    away_events = team_events[feature_cols].rename(columns={
        'team': 'away_team',
        'rolling_mean_goals_for': 'away_rolling_mean_goals_for',
        'rolling_mean_goals_against': 'away_rolling_mean_goals_against',
        'days_since_last': 'away_days_since_last',
        'matches_last_14d': 'away_matches_last_14d',
    })
    matches_df = matches_df.merge(away_events, how='left', on=['away_team', 'date'])

    return matches_df


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
    team_events['rolling_mean_goals_for'] = team_events.groupby('team')['goals_for'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    team_events['rolling_mean_goals_against'] = team_events.groupby('team')['goals_against'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    team_events['days_since_last'] = team_events.groupby('team')['date'].diff().dt.days

    team_events['matches_last_14d'] = team_events.groupby('team', group_keys=False).apply(
        lambda x: x.set_index('date')['team'].rolling('14D', closed='left').count()
    ).reset_index(level=0, drop=True).fillna(0).values

    return team_events


if __name__ == '__main__':
    matches = load_football_data_csv(Path('../data/E0.csv'), '24/25')
    team_events = build_team_events(matches)
    matches = merge_team_events(matches, team_events)

    matches.to_csv('../data/test.csv', index=False)
    print(matches.head(20))
