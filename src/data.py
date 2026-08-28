from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA_COLUMN_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",

    # Shots on target is a lower-variance measure of the same attacking strength goals measure:
    # ~4.8 vs ~1.6 per team per game, so a 5-match window rests on ~24 events instead of ~8.
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
}

# football-data renamed its aggregate odds columns for 2019/20 (Betbrain Bb* -> Max*/Avg*).
# Pre-2019/20 files have no closing-average columns, and Pinnacle (PS*) only starts in 2012/13.
# First alias present wins; a role with no alias present becomes an all-NaN column so callers
# can detect it rather than silently mis-key.
ODDS_ALIASES = {
    # Average across books (~4.5% overround) -- the devig anchor for base_margin.
    "decision_odds_home": ["AvgH", "BbAvH"],
    "decision_odds_draw": ["AvgD", "BbAvD"],
    "decision_odds_away": ["AvgA", "BbAvA"],

    # Best price across books (~1% overround). A better *price*, not a better *probability*.
    "best_odds_home": ["MaxH", "BbMxH"],
    "best_odds_draw": ["MaxD", "BbMxD"],
    "best_odds_away": ["MaxA", "BbMxA"],

    # Pinnacle: sharpest single book (~3.6%), tolerates winning accounts. Absent before 2012/13.
    "pinnacle_odds_home": ["PSH"],
    "pinnacle_odds_draw": ["PSD"],
    "pinnacle_odds_away": ["PSA"],

    "closing_odds_home": ["AvgCH"],
    "closing_odds_draw": ["AvgCD"],
    "closing_odds_away": ["AvgCA"],
}

# Metrics rolled into venue-split form features. Adding one here (plus home_/away_ entries in
# SCHEMA_COLUMN_MAP) flows through build_team_events and merge_team_events automatically.
ROLLING_METRICS = ['goals', 'shots', 'shots_on_target']
ROLLING_FEATURES = [f'{m}_{side}' for m in ROLLING_METRICS for side in ('for', 'against')]

def merge_team_events(matches_df: pd.DataFrame, team_events: pd.DataFrame) -> pd.DataFrame:
    recency_cols = ['team', 'date', 'days_since_last', 'matches_last_14d']

    home_recency = team_events[recency_cols].rename(columns={
        'team': 'home_team',
        'days_since_last': 'home_days_since_last',
        'matches_last_14d': 'home_matches_last_14d',
    })
    matches_df = matches_df.merge(home_recency, how='left', on=['home_team', 'date'])

    away_recency = team_events[recency_cols].rename(columns={
        'team': 'away_team',
        'days_since_last': 'away_days_since_last',
        'matches_last_14d': 'away_matches_last_14d',
    })
    matches_df = matches_df.merge(away_recency, how='left', on=['away_team', 'date'])

    rolled = [f'rolling_mean_{c}' for c in ROLLING_FEATURES]
    form_cols = ['team', 'date'] + rolled

    home_form = team_events[team_events['venue'] == 'home'][form_cols].rename(columns={
        'team': 'home_team', **{c: f'home_{c}' for c in rolled},
    })
    matches_df = matches_df.merge(home_form, how='left', on=['home_team', 'date'])

    away_form = team_events[team_events['venue'] == 'away'][form_cols].rename(columns={
        'team': 'away_team', **{c: f'away_{c}' for c in rolled},
    })
    matches_df = matches_df.merge(away_form, how='left', on=['away_team', 'date'])

    return matches_df


def load_football_data_csv(path: Path, season: str):
    raw = pd.read_csv(path)
    # Some files carry a trailing blank row (e.g. E01415.csv); drop anything without a fixture.
    raw = raw.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
    raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=True)

    core = raw[list(SCHEMA_COLUMN_MAP.keys())].rename(columns=SCHEMA_COLUMN_MAP)

    odds = {}
    for role, aliases in ODDS_ALIASES.items():
        present = next((a for a in aliases if a in raw.columns), None)
        odds[role] = raw[present] if present else np.nan

    out = pd.concat([core, pd.DataFrame(odds, index=raw.index)], axis=1)
    out["season"] = season
    return out


def load_seasons(data_dir: Path, filenames_and_seasons) -> pd.DataFrame:
    """Loads and concatenates multiple seasons' CSVs into one date-sorted frame. Rolling/recency
    features (build_team_events) then naturally carry a promoted/relegated team's missing prior
    top-flight history as NaN, and a returning team's rolling form across the summer break.
    """
    frames = [load_football_data_csv(data_dir / fname, season) for fname, season in filenames_and_seasons]
    return pd.concat(frames, ignore_index=True).sort_values('date').reset_index(drop=True)


def build_team_events(matches_df: pd.DataFrame):
    # Each metric becomes a for/against pair per team-match, rolled identically.
    home_events = matches_df[['date', 'home_team'] + [f'{s}_{m}' for m in ROLLING_METRICS
                                                      for s in ('home', 'away')]].rename(
        columns={'home_team': 'team',
                 **{f'home_{m}': f'{m}_for' for m in ROLLING_METRICS},
                 **{f'away_{m}': f'{m}_against' for m in ROLLING_METRICS}}
    )
    home_events['venue'] = 'home'
    away_events = matches_df[['date', 'away_team'] + [f'{s}_{m}' for m in ROLLING_METRICS
                                                      for s in ('home', 'away')]].rename(
        columns={'away_team': 'team',
                 **{f'away_{m}': f'{m}_for' for m in ROLLING_METRICS},
                 **{f'home_{m}': f'{m}_against' for m in ROLLING_METRICS}}
    )
    away_events['venue'] = 'away'

    team_events = pd.concat([home_events, away_events])
    team_events = team_events.sort_values(by=['team', 'date']).reset_index(drop=True)

    # shift(1) before rolling so a match never sees its own result; grouped by venue so "home
    # form" only looks at previous home matches.
    for col in ROLLING_FEATURES:
        team_events[f'rolling_mean_{col}'] = team_events.groupby(['team', 'venue'])[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    team_events['days_since_last'] = team_events.groupby('team')['date'].diff().dt.days

    team_events['_match_count_helper'] = 1
    result = team_events.groupby('team').rolling('14D', on='date', closed='left')['_match_count_helper'].count()
    team_events['matches_last_14d'] = result.fillna(0).values
    team_events = team_events.drop(columns='_match_count_helper')

    return team_events


if __name__ == '__main__':
    matches = load_football_data_csv(Path('../data/E0.csv'), '24/25')
    team_events = build_team_events(matches)
    matches = merge_team_events(matches, team_events)

    matches.to_csv('../data/test.csv', index=False)
    print(matches.head(20))
