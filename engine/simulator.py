"""
simulator.py — Monte Carlo Conference Winner Simulator

Draws team strengths from the BTD posterior and simulates remaining
conference games to estimate conference regular-season winner probabilities.

Can also be run in "full season" mode to simulate the entire conference
season from scratch (ignoring current standings) for backtesting.

Usage:
    python simulator.py                          # 10k sims, all conferences
    python simulator.py --sims 50000             # more simulations
    python simulator.py --conference "Hockey East" # single conference
"""

import argparse
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load conference and tournament configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


# Load config at module level
_CONFIG = load_config()
CONFERENCES = _CONFIG["conferences"]

# note: independents are not included in conference simulations

# softmax outcome parameters
# matches btd.py: RW=0, OW=1, T=2, OL=3, RL=4
P_COEFS = np.array([1.0, 2/3, 1/2, 1/3, 0.0])
O_COEFS = np.array([0.0, 1.0, 1.0, 1.0, 0.0])

# conference points: RW=3, OW=2, T=1, OL=1, RL=0
POINTS = np.array([3, 2, 1, 1, 0])


def load_posterior(path: str) -> dict:
    """Load posterior samples from btd.py output."""
    data = np.load(path, allow_pickle=True)
    teams = list(data["teams"])
    team_to_idx = {t: i for i, t in enumerate(teams)}
    return {
        "lam_samples": data["lam_samples"],  # (n_draws, n_teams)
        "tau_samples": data["tau_samples"],   # (n_draws,)
        "teams": teams,
        "team_to_idx": team_to_idx,
    }


def softmax_probs(gamma: float, tau: float) -> np.ndarray:
    """
    Compute outcome probabilities for a single game.

    Args:
        gamma: lambda_home - lambda_away (strength difference)
        tau: overtime parameter

    Returns:
        Array of 5 probabilities: [P(RW), P(OW), P(T), P(OL), P(RL)]
    """
    scores = P_COEFS * gamma + O_COEFS * tau
    # numerically stable softmax
    scores -= scores.max()
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()


def simulate_game(gamma: float, tau: float, rng: np.random.Generator) -> int:
    """Simulate a single game, return outcome index (0-4)."""
    probs = softmax_probs(gamma, tau)
    return rng.choice(5, p=probs)


def generate_round_robin(teams: list) -> list:
    """
    Generate a full home-and-away round robin schedule.
    Returns list of (home, away) tuples.
    """
    games = []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i != j:
                games.append((home, away))
    return games


def simulate_conference(
    conf_name: str,
    conf_teams: list,
    posterior: dict,
    n_sims: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate a conference season n_sims times.

    For each simulation:
      1. Draw team strengths from posterior
      2. Play full round-robin (home and away)
      3. Record conference points
      4. Track who finishes #1
    """
    team_to_idx = posterior["team_to_idx"]
    lam_samples = posterior["lam_samples"]
    tau_samples = posterior["tau_samples"]
    n_draws = lam_samples.shape[0]

    # conference team indices into the global team array
    conf_idx = [team_to_idx[t] for t in conf_teams]

    # generate round-robin schedule
    schedule = generate_round_robin(conf_teams)

    # track results
    win_counts = defaultdict(int)         # times each team finishes #1
    total_points = defaultdict(list)      # points per sim for each team
    avg_rank = defaultdict(float)         # running rank sum

    for _ in range(n_sims):
        # draw from posterior (random row)
        draw_idx = rng.integers(0, n_draws)
        lam = lam_samples[draw_idx]
        tau = tau_samples[draw_idx]

        # simulate all conference games
        points = {t: 0 for t in conf_teams}

        for home, away in schedule:
            h_idx = team_to_idx[home]
            a_idx = team_to_idx[away]
            gamma = lam[h_idx] - lam[a_idx]

            outcome = simulate_game(gamma, tau, rng)

            # home team gets POINTS[outcome], away gets POINTS[4-outcome]
            # bc outcomes are symmetric: RW↔RL, OW↔OL, T↔T
            points[home] += POINTS[outcome]
            points[away] += POINTS[4 - outcome]

        # rank teams by points (tiebreak: random for now)
        ranked = sorted(conf_teams, key=lambda t: (points[t], rng.random()), reverse=True)

        # record
        win_counts[ranked[0]] += 1
        for rank, team in enumerate(ranked, 1):
            total_points[team].append(points[team])
            avg_rank[team] += rank

    # build results DataFrame
    rows = []
    for team in conf_teams:
        pts = total_points[team]
        rows.append({
            "Conference": conf_name,
            "Team": team,
            "Win_Conf_%": win_counts[team] / n_sims * 100,
            "Avg_Points": np.mean(pts),
            "Avg_Rank": avg_rank[team] / n_sims,
            "P_Top4_%": sum(1 for p in range(n_sims)
                           if total_points[team][p] >= sorted(
                               [total_points[t][p] for t in conf_teams],
                               reverse=True)[3]) / n_sims * 100,
        })

    df = pd.DataFrame(rows).sort_values("Win_Conf_%", ascending=False)
    return df


# ── NCAA Tournament bracket ─────────────────────────────────────────
# Loaded from config.json. Each regional: [1-seed, 4-seed, 2-seed, 3-seed]
# Semis: 1v4 and 2v3, then winners play regional final.
# Frozen Four: Regional 1 winner vs Regional 2 winner,
#              Regional 3 winner vs Regional 4 winner, then championship.

def _parse_tournament_bracket(config: dict) -> dict:
    """Convert config bracket format to tuple format used by simulator."""
    bracket = {}
    for region, teams in config["tournament"]["bracket"].items():
        bracket[region] = [(t["team"], t["seed"]) for t in teams]
    return bracket

def _parse_frozen_four_matchups(config: dict) -> list:
    """Convert config frozen four matchups to tuple format."""
    return [tuple(m) for m in config["tournament"]["frozen_four_matchups"]]

TOURNAMENT_BRACKET = _parse_tournament_bracket(_CONFIG)
FROZEN_FOUR_MATCHUPS = _parse_frozen_four_matchups(_CONFIG)


def simulate_tournament_game(
    team_a: str,
    team_b: str,
    lam: np.ndarray,
    tau: float,
    team_to_idx: dict,
    rng: np.random.Generator,
) -> str:
    """
    Simulate a single-elimination tournament game.

    NCAA tournament games CANNOT end in a tie — they play OT periods
    until someone scores. So we collapse the 5-outcome model:
      P(A wins) = P(RW) + P(OW) + P(T)/2
      P(B wins) = P(RL) + P(OL) + P(T)/2

    The T/2 split reflects that a tie goes to sudden-death OT where
    each team has ~50% chance (already evenly matched if it's tied).

    Returns the winning team name.
    """
    a_idx = team_to_idx[team_a]
    b_idx = team_to_idx[team_b]
    gamma = lam[a_idx] - lam[b_idx]

    probs = softmax_probs(gamma, tau)
    # probs = [P(RW), P(OW), P(T), P(OL), P(RL)]
    p_a_wins = probs[0] + probs[1] + probs[2] / 2
    # p_b_wins = probs[3] + probs[4] + probs[2] / 2

    if rng.random() < p_a_wins:
        return team_a
    else:
        return team_b


def simulate_tournament(
    posterior: dict,
    n_sims: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate the NCAA tournament n_sims times.

    For each simulation:
      1. Draw team strengths from posterior
      2. Play each regional (semis + final)
      3. Play Frozen Four (semis + championship)
      4. Track advancement counts
    """
    team_to_idx = posterior["team_to_idx"]
    lam_samples = posterior["lam_samples"]
    tau_samples = posterior["tau_samples"]
    n_draws = lam_samples.shape[0]

    # all tournament teams
    all_teams = []
    team_seeds = {}
    for region, teams in TOURNAMENT_BRACKET.items():
        for team, seed in teams:
            all_teams.append(team)
            team_seeds[team] = seed

    # track advancement
    make_regional_final = defaultdict(int)  
    make_frozen_four = defaultdict(int)
    make_championship = defaultdict(int)
    win_championship = defaultdict(int)

    for _ in range(n_sims):
        draw_idx = rng.integers(0, n_draws)
        lam = lam_samples[draw_idx]
        tau = tau_samples[draw_idx]

        sim_game = lambda a, b: simulate_tournament_game(
            a, b, lam, tau, team_to_idx, rng
        )

        # play regionals
        regional_winners = {}

        for region_name, teams in TOURNAMENT_BRACKET.items():
            # semis: 1-seed vs 4-seed, 2-seed vs 3-seed
            # teams is [(team, seed), ...] ordered: 1, 4, 2, 3
            semi1_winner = sim_game(teams[0][0], teams[1][0])
            semi2_winner = sim_game(teams[2][0], teams[3][0])

            make_regional_final[semi1_winner] += 1
            make_regional_final[semi2_winner] += 1

            # regional final
            regional_winner = sim_game(semi1_winner, semi2_winner)
            regional_winners[region_name] = regional_winner
            make_frozen_four[regional_winner] += 1

        # frozen four
        ff_r1, ff_r2 = FROZEN_FOUR_MATCHUPS[0]
        ff_r3, ff_r4 = FROZEN_FOUR_MATCHUPS[1]

        semi1_winner = sim_game(regional_winners[ff_r1], regional_winners[ff_r2])
        semi2_winner = sim_game(regional_winners[ff_r3], regional_winners[ff_r4])

        make_championship[semi1_winner] += 1
        make_championship[semi2_winner] += 1

        # championship
        champion = sim_game(semi1_winner, semi2_winner)
        win_championship[champion] += 1

    # build results
    rows = []
    for team in all_teams:
        rows.append({
            "Team": team,
            "Seed": team_seeds[team],
            "Regional_Final_%": make_regional_final[team] / n_sims * 100,
            "Frozen_Four_%": make_frozen_four[team] / n_sims * 100,
            "Championship_Game_%": make_championship[team] / n_sims * 100,
            "Win_Title_%": win_championship[team] / n_sims * 100,
        })

    df = pd.DataFrame(rows).sort_values("Win_Title_%", ascending=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo conference & tournament simulator")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations")
    parser.add_argument("--posterior", default="../engine/posterior_samples.npz",
                        help="Path to posterior samples")
    parser.add_argument("--conference", default=None,
                        help="Simulate a single conference (default: all)")
    parser.add_argument("--tournament", action="store_true",
                        help="Simulate the NCAA tournament")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # load posterior
    posterior = load_posterior(args.posterior)
    print(f"Loaded {posterior['lam_samples'].shape[0]} posterior draws for "
          f"{len(posterior['teams'])} teams")

    rng = np.random.default_rng(args.seed)

    # NCAA tournament simulations
    if args.tournament:
        df = simulate_tournament(posterior, args.sims, rng)

        print(f"\n{'':2s}{'Team':25s}  {'Seed':>4s}  {'Reg Final':>9s}  "
              f"{'Frozen 4':>8s}  {'Title Game':>10s}  {'Champion':>8s}")
        for _, row in df.iterrows():
            print(f"  {row['Team']:25s}  {row['Seed']:4.0f}  "
                  f"{row['Regional_Final_%']:8.1f}%  {row['Frozen_Four_%']:7.1f}%  "
                  f"{row['Championship_Game_%']:9.1f}%  {row['Win_Title_%']:7.1f}%")

        df.to_csv("tournament_probabilities.csv", index=False)
        print(f"\nSaved tournament_probabilities.csv")
        return

    # conference simulations
    if args.conference:
        confs = {args.conference: CONFERENCES[args.conference]}
    else:
        confs = CONFERENCES

    all_results = []
    for conf_name, conf_teams in confs.items():
        df = simulate_conference(conf_name, conf_teams, posterior, args.sims, rng)
        all_results.append(df)

        # Print results
        print(f"\n{'':2s}{'Team':25s}  {'Win Conf%':>9s}  {'Avg Pts':>7s}  {'Avg Rank':>8s}  {'Top 4%':>6s}")
        for _, row in df.iterrows():
            print(f"  {row['Team']:25s}  {row['Win_Conf_%']:8.1f}%  {row['Avg_Points']:7.1f}  "
                  f"{row['Avg_Rank']:8.02f}  {row['P_Top4_%']:5.1f}%")

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv("conference_probabilities.csv", index=False)
    print(f"\nSaved conference_probabilities.csv ({len(results)} rows)")


if __name__ == "__main__":
    main()