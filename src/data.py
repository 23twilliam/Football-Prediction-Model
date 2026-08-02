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


if __name__ == '__main__':
    dataframe = load_football_data_csv(Path('../data/E0.csv'), '24/25')
    print(dataframe.head())
