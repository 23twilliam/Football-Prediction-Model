import pandas as pd


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

def kelly_stake(model_prob, decimal_odds, fraction: float= 0.25, cap = 0.025):
    b = decimal_odds - 1 # Net odds, profit per unit staked if win
    edge = model_prob * decimal_odds - 1
    if edge <= 0:
        return 0.0
    else:
        full_kelly = edge / b
        conservative_kelly = full_kelly * fraction # Quarter by default
        return min(conservative_kelly, cap)