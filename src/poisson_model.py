import numpy as np
import scipy.stats as stats
import pandas as pd
import scipy.optimize as optimize


def tau_correction(x, y, lam, mu, rho):
    tau = np.ones_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    m10 = (x == 1) & (y == 0)

    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + lam[m10] * rho
    tau[m11] = 1 - rho

    return tau


def fit_attack_defence(matches_df: pd.DataFrame, l2_reg: int = 0.02):
    teams = sorted(pd.concat([matches_df["home_team"], matches_df["away_team"]]).unique())
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    # Converts Name into index to help with later we can easily see each teams attack rating without string comparison
    home_idx = matches_df['home_team'] = matches_df['home_team'].map(team_to_idx)
    away_idx = matches_df['away_team'] = matches_df['away_team'].map(team_to_idx)

    home_goals = matches_df['home_goals']
    away_goals = matches_df['away_goals']

    # Defined INSIDE since scipy.optimize.minimize will only ever call this as nll(params)
    def nll(params):
        # vector of attack, defence, home_adv, rho
        attack = params[:n_teams]
        defence = params[n_teams: 2 * n_teams]
        home_adv = params[-2]
        rho = params[-1]

        # Exp() because of the Log-Linear parameterisation, guarantees lam/mu stay positive no matter what the numbers
        # happen to be
        lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)  # Home attack vs Away Defence (+ home adv)
        mu = np.exp(attack[away_idx] - defence[home_idx])  # Opposite

        # Joint probability of the exact observed scoreline for every match. Same formula as poisson, but vectorised
        # across every match's own lam/mu instead of one shared league wide pair
        p = (stats.poisson.pmf(home_goals, lam) * stats.poisson.pmf(away_goals, mu)
             * tau_correction(home_goals, away_goals, lam, mu, rho))
        # Stops log(0)
        p = np.clip(p, 1e-10, None)

        # L2 penalty = a Gaussian prior (mean 0) on every attack/defence value, in MAP-estimation terms. Stops ratings
        # from diverging to extreme values on sparse data
        penalty = l2_reg * (np.sum(attack ** 2) + np.sum(defence ** 2))

        # Negative log likelihood calculation
        return -np.sum(np.log(p)) + penalty

    # Starting guess, all teams begin "equal" (zero attack, or defence) with a small nudge for home advantage,
    # must be same length as what nll expects above
    x0 = np.concatenate([np.zeros(n_teams), np.zeros(n_teams), [0.065], [0.0]])

    # Bounds for extremes
    bounds = [(None, None)] * (2 * n_teams) + [(None, None)], (-0.3, 0.3)

    # Walks until nll's gradient is zero.
    result = optimize.minimize(nll, x0, method='L-BFGS-B', bounds=bounds)  # quasi-Newton method, uses the gradient of
    # log likelihood.

    # Unpack with same slicing as nll
    attack = result.x[:n_teams]
    defence = result.x[n_teams:2 * n_teams]
    home_adv = result.x[-2]
    rho = result.x[-1]

    return teams, attack, defence, home_adv, rho


def simple_match_probs(lam: float, mu: float, max_goals: int = 10) -> dict:
    home_wins = 0
    draws = 0
    away_wins = 0
    for x in range(max_goals):
        for y in range(max_goals):
            joint_p = stats.poisson.pmf(x, lam) * stats.poisson.pmf(y, mu)
            if x > y:
                home_wins += joint_p
            elif x == y:
                draws += joint_p
            else:
                away_wins += joint_p

    return {'Home': home_wins, 'Draw': draws, 'Away': away_wins}
