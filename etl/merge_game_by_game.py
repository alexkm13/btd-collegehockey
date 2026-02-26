import pandas as pd
from team_mapping import ESPN_TO_CHN


def load_and_clean_games(path: str) -> pd.DataFrame:
    """Load ESPN game results, map team names, drop bad records."""
    df = pd.read_csv(path)

    # Map ESPN names → CHN names
    df["home_team"] = df["home_team"].map(ESPN_TO_CHN)
    df["away_team"] = df["away_team"].map(ESPN_TO_CHN)

    # Drop games involving non-D1 teams (Assumption, etc.)
    df = df.dropna(subset=["home_team", "away_team"])

    n_before = len(df)

    # Drop period=0 games (bad data: 0-0 scores marked Final)
    df = df[df["period"] > 0]

    n_dropped = n_before - len(df)
    print(f"Dropped {n_dropped} bad records (period=0, no score data)")

    # Reclassify: only keep RW, RL, OW, OL, T
    # Verify all outcomes are in expected set
    valid = {"RW", "RL", "OW", "OL", "T"}
    assert set(df["home_outcome"].unique()) <= valid, f"Unexpected outcomes: {set(df['home_outcome'].unique()) - valid}"

    print(f"Clean games: {len(df)}")
    print(f"Outcome distribution:\n{df['home_outcome'].value_counts().to_string()}")
    print(f"Unique teams: {len(set(df['home_team']) | set(df['away_team']))}")

    return df


def load_covariates(path: str) -> pd.DataFrame:
    """Load CHN combined stats, extract the three covariates."""
    df = pd.read_csv(path)
    covariates = df[["Team", "Close_FF%", "PP%", "GSAx_60"]].copy()
    print(f"Loaded covariates for {len(covariates)} teams")
    return covariates


def main():
    # Load and clean
    games = load_and_clean_games("data/espn_game_results.csv")
    covariates = load_covariates("data/chn_combined.csv")

    # Verify all teams in games exist in covariates
    game_teams = set(games["home_team"]) | set(games["away_team"])
    cov_teams = set(covariates["Team"])
    missing = game_teams - cov_teams
    if missing:
        print(f"WARNING: Teams in games but not in covariates: {missing}")
    else:
        print("All game teams have covariates ✓")

    # Save game results (BTD model input)
    # Keep only the columns the model needs
    game_cols = ["date", "home_team", "away_team", "home_score", "away_score",
                 "period", "home_outcome", "away_outcome"]
    games[game_cols].to_csv("game_results.csv", index=False)
    print(f"\nSaved game_results.csv ({len(games)} games)")

    # Save team covariates (BTD covariate input)
    covariates.to_csv("team_covariates.csv", index=False)
    print(f"Saved team_covariates.csv ({len(covariates)} teams)")


if __name__ == "__main__":
    main()