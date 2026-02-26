"""
Covariate Stickiness Analysis

Computes year-over-year Pearson correlations for each covariate
to determine which capture durable program traits vs. transient roster quality.

Usage:
    python stickiness.py

Input:
    data/covariates_all.csv  (from scrape_historical.py)

Output:
    Prints correlation matrix and scatter plots
    data/stickiness_results.csv
"""

import pandas as pd
import numpy as np
from scipy import stats


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/covariates_all.csv")
    print(f"Loaded {len(df)} rows across seasons: {df['season'].unique()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Teams per season:")
    print(df.groupby("season")["Team"].count())
    print()
    return df


def compute_stickiness(df: pd.DataFrame, covariate: str) -> pd.DataFrame:
    """
    For each consecutive season pair, merge teams and compute Pearson r
    for the given covariate.
    """
    seasons = sorted(df["season"].unique())
    results = []

    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        df1 = df[df["season"] == s1][["Team", covariate]].rename(
            columns={covariate: f"{covariate}_prev"}
        )
        df2 = df[df["season"] == s2][["Team", covariate]].rename(
            columns={covariate: f"{covariate}_curr"}
        )

        merged = df1.merge(df2, on="Team", how="inner")
        merged = merged.dropna()

        if len(merged) < 10:
            print(f"  WARNING: Only {len(merged)} teams matched for {s1} -> {s2}")
            continue

        r, p = stats.pearsonr(
            merged[f"{covariate}_prev"], merged[f"{covariate}_curr"]
        )

        results.append({
            "covariate": covariate,
            "season_pair": f"{s1} -> {s2}",
            "n_teams": len(merged),
            "pearson_r": round(r, 3),
            "p_value": round(p, 4),
            "r_squared": round(r ** 2, 3),
        })

        print(f"  {s1} -> {s2}: r = {r:.3f} (p = {p:.4f}, n = {len(merged)})")

    return pd.DataFrame(results)


def main():
    df = load_data()

    covariates = ["FF%_close", "PP%", "GSAx/60"]
    all_results = []

    for cov in covariates:
        if cov not in df.columns or df[cov].isna().all():
            print(f"\nSKIPPING {cov} -- no data available")
            print(f"  (You may need to fix column mapping in scrape_historical.py)")
            continue

        print(f"\n{'='*50}")
        print(f"Stickiness: {cov}")
        print(f"{'='*50}")

        results = compute_stickiness(df, cov)
        all_results.append(results)

    if all_results:
        df_results = pd.concat(all_results, ignore_index=True)
        print(f"\n\n{'='*60}")
        print("STICKINESS SUMMARY")
        print(f"{'='*60}")
        print(df_results.to_string(index=False))

        # Average r across season pairs per covariate
        print(f"\nAverage Pearson r by covariate:")
        avg = df_results.groupby("covariate")["pearson_r"].mean()
        for cov, r in avg.items():
            label = "STICKY" if abs(r) > 0.4 else "MODERATE" if abs(r) > 0.2 else "NOT STICKY"
            print(f"  {cov:15s}: r = {r:.3f}  [{label}]")

        df_results.to_csv("data/stickiness_results.csv", index=False)
        print(f"\nSaved: data/stickiness_results.csv")

        # Interpretation for the paper
        print(f"\n{'='*60}")
        print("INTERPRETATION FOR PAPER")
        print(f"{'='*60}")
        print("""
If FF%_close r > 0.5:
  → Possession quality is a durable program-level trait.
  → Safe to use cross-season for prediction.
  → Reflects coaching system, recruiting pipeline.

If PP% r ~ 0.3-0.5:
  → Moderate stickiness. Special teams partially system,
    partially personnel. Useful but noisier cross-season.

If GSAx/60 r < 0.2:
  → Goaltending is roster-dependent, not program-dependent.
  → Last year's GSAx tells you little about this year.
  → Including stale GSAx may ADD NOISE to predictions.
  → Finding: "sticky-only" model may beat "full" model.
        """)


if __name__ == "__main__":
    main()