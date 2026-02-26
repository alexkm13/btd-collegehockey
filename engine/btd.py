"""
btd.py — Covariate-Enhanced Bayesian Bradley-Terry-Davidson Model

Extends Whelan & Klein (2021) with three covariates:
  - FF% close (possession quality)
  - PP% (power play %)
  - GSAx/60 ES (goaltending quality)

Five outcomes: RW, OW, T, OL, RL (extends Whelan's 4-outcome to handle NCAA ties)

Softmax formulation:
  P(outcome I | i vs j) = exp(x^I) / sum_J exp(x^J)
  where x^I = p^I * gamma_ij + o^I * tau

  gamma_ij = lambda_i - lambda_j  (strength difference)
  lambda_i = alpha_i + beta_1 * FF%_i + beta_2 * PP%_i + beta_3 * GSAx_i

  Outcome parameters (extending Whelan Table 1 + Davidson tie):
    RW: p=1,   o=0   (full win for i)
    OW: p=2/3, o=1   (partial win, overtime)
    T:  p=1/2, o=1   (draw, overtime)
    OL: p=1/3, o=1   (partial loss, overtime)
    RL: p=0,   o=0   (full loss for i)

Usage:
    python btd.py              # Fit model, print summary
    python btd.py --samples N  # Custom number of posterior samples
"""

import argparse
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


# maps outcome string → integer index for the categorical likelihood
OUTCOME_MAP = {"RW": 0, "OW": 1, "T": 2, "OL": 3, "RL": 4}

# softmax score coefficients per outcome (from home team i's perspective)
# x^I = p^I * gamma_ij + o^I * tau
P_COEFS = np.array([1.0, 2 / 3, 1 / 2, 1 / 3, 0.0])  # strength weight
O_COEFS = np.array([0.0, 1.0, 1.0, 1.0, 0.0])  # overtime flag


def load_data(games_path: str, covariates_path: str):
    """Load game results and team covariates, build index mappings."""
    games = pd.read_csv(games_path)
    covariates = pd.read_csv(covariates_path)

    # Build team → index mapping (sorted alphabetically for reproducibility)
    teams = sorted(covariates["Team"].unique())
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Encode game data as integer arrays
    home_idx = games["home_team"].map(team_to_idx).values.astype(int)
    away_idx = games["away_team"].map(team_to_idx).values.astype(int)
    outcome_idx = games["home_outcome"].map(OUTCOME_MAP).values.astype(int)

    # Build covariate matrix (n_teams x 3), standardized
    cov_cols = ["Close_FF%", "PP%", "GSAx_60"]
    X_raw = covariates.set_index("Team").loc[teams, cov_cols].values.astype(float)

    # Standardize: zero mean, unit variance (helps sampling + interpretation)
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X = (X_raw - X_mean) / X_std

    print(f"Loaded {len(games)} games, {n_teams} teams, {len(cov_cols)} covariates")
    print(f"Outcome distribution: {dict(games['home_outcome'].value_counts())}")
    print(f"Covariate means: {dict(zip(cov_cols, X_mean.round(2)))}")
    print(f"Covariate stds:  {dict(zip(cov_cols, X_std.round(2)))}")

    return {
        "home_idx": home_idx,
        "away_idx": away_idx,
        "outcome_idx": outcome_idx,
        "X": X,
        "X_mean": X_mean,
        "X_std": X_std,
        "n_teams": n_teams,
        "n_games": len(games),
        "teams": teams,
        "team_to_idx": team_to_idx,
        "cov_cols": cov_cols,
    }


def build_model(data: dict) -> pm.Model:
    """
    Build the PyMC BTD model.

    Model structure:
      alpha ~ Normal(0, 1)         [n_teams - 1 free params, last team = -sum]
      beta  ~ Normal(0, 1)         [3 covariate weights]
      tau   ~ Normal(0, 1)         [overtime parameter]

      lambda_i = alpha_i + X_i @ beta
      gamma_ij = lambda_i - lambda_j
      score_I  = p_I * gamma_ij + o_I * tau

      outcome ~ Categorical(softmax(scores))
    """
    n_teams = data["n_teams"]
    X = data["X"]  # (n_teams, 3) standardized covariates
    home_idx = data["home_idx"]  # (n_games,)
    away_idx = data["away_idx"]  # (n_games,)
    outcome_idx = data["outcome_idx"]  # (n_games,)

    with pm.Model() as model:
        # ── Priors ──────────────────────────────────────────────
        # Team-specific intercepts (sum-to-zero constraint via n-1 parameterization)
        alpha_free = pm.Normal("alpha_free", mu=0, sigma=1, shape=n_teams - 1)
        # Last team's alpha = negative sum of all others (enforces sum-to-zero)
        alpha = pt.concatenate([alpha_free, -pt.sum(alpha_free, keepdims=True)])

        # Covariate weights
        beta = pm.Normal("beta", mu=0, sigma=1, shape=3)

        # Overtime parameter (log-scale: tau = ln(nu) in Whelan notation)
        tau = pm.Normal("tau", mu=0, sigma=1)

        # ── Team strengths ──────────────────────────────────────
        # lambda_i = alpha_i + X_i @ beta
        lam = alpha + pt.dot(pt.as_tensor_variable(X), beta)

        # Store as deterministic for easy posterior extraction
        pm.Deterministic("lambda", lam)

        # ── Game-level softmax ──────────────────────────────────
        # Strength difference for each game
        gamma = lam[home_idx] - lam[away_idx]  # (n_games,)

        # Compute softmax scores for all 5 outcomes
        # scores[g, I] = p_I * gamma[g] + o_I * tau
        p_coefs = pt.as_tensor_variable(P_COEFS)  # (5,)
        o_coefs = pt.as_tensor_variable(O_COEFS)  # (5,)

        # Broadcast: gamma is (n_games,), p_coefs is (5,)
        # Result: (n_games, 5)
        scores = gamma[:, None] * p_coefs[None, :] + tau * o_coefs[None, :]

        # ── Likelihood ──────────────────────────────────────────
        # Custom categorical: pick the log-prob of the observed outcome
        pm.Categorical("obs", p=pm.math.softmax(scores, axis=1), observed=outcome_idx)

    return model


def fit_model(model: pm.Model, samples: int = 2000, chains: int = 4, cores: int = 4):
    """Run MCMC sampling with NUTS (PyMC's default HMC variant)."""
    with model:
        trace = pm.sample(
            draws=samples,
            tune=1000,
            chains=chains,
            cores=cores,
            target_accept=0.9,  # higher acceptance for complex models
            random_seed=42,
            return_inferencedata=True,
        )
    return trace


def summarize_results(trace, data: dict):
    """Print model results: team rankings, covariate effects, diagnostics."""
    import arviz as az

    teams = data["teams"]
    cov_cols = data["cov_cols"]
    X_std = data["X_std"]

    # diagnostics
    print("\nDIAGNOSTICS")

    # check R-hat and divergences
    summary = az.summary(trace, var_names=["alpha_free", "beta", "tau"])
    rhat_max = summary["r_hat"].max()
    ess_min = summary["ess_bulk"].min()
    n_div = trace.sample_stats.diverging.sum().values
    print(f"Max R-hat: {rhat_max:.4f} (want ≈ 1.0)")
    print(f"Min ESS:   {ess_min:.0f} (want > 400)")
    print(f"Divergences: {n_div} (want 0)")

    # covariate effects
    print("\nCOVARIATE EFFECTS (standardized)")

    beta_samples = trace.posterior["beta"].values.reshape(-1, 3)
    for i, col in enumerate(cov_cols):
        mean = beta_samples[:, i].mean()
        sd = beta_samples[:, i].std()
        p_pos = (beta_samples[:, i] > 0).mean()
        # convert to original scale: beta_orig = beta_std / X_std
        beta_orig = mean / X_std[i]
        print(
            f"  {col:12s}: β = {mean:+.3f} ± {sd:.3f}  (P(β>0) = {p_pos:.1%})  "
            f"[per 1 unit orig: {beta_orig:+.4f}]"
        )

    # overtime parameter
    tau_samples = trace.posterior["tau"].values.flatten()
    print(f"\n  {'tau':12s}: τ = {tau_samples.mean():+.3f} ± {tau_samples.std():.3f}")

    # team rankings
    print("\nTEAM RANKINGS (by posterior mean λ)")

    lam_samples = trace.posterior["lambda"].values.reshape(-1, data["n_teams"])
    lam_mean = lam_samples.mean(axis=0)
    lam_sd = lam_samples.std(axis=0)

    # sort by mean strength
    ranking = sorted(range(data["n_teams"]), key=lambda i: -lam_mean[i])

    print(f"{'Rank':>4s}  {'Team':25s}  {'λ mean':>8s}  {'λ sd':>6s}  {'95% CI':>16s}")
    for rank, i in enumerate(ranking, 1):
        ci_lo = np.percentile(lam_samples[:, i], 2.5)
        ci_hi = np.percentile(lam_samples[:, i], 97.5)
        print(
            f"{rank:4d}  {teams[i]:25s}  {lam_mean[i]:+8.3f}  {lam_sd[i]:6.3f}  "
            f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
        )

    return {
        "lam_samples": lam_samples,
        "beta_samples": beta_samples,
        "tau_samples": tau_samples,
        "lam_mean": lam_mean,
        "lam_sd": lam_sd,
        "ranking": ranking,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit BTD model")
    parser.add_argument(
        "--samples", type=int, default=2000, help="Posterior samples per chain"
    )
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains")
    parser.add_argument(
        "--games",
        default="../etl/data/game_results.csv",
        help="Path to game results CSV",
    )
    parser.add_argument(
        "--covariates",
        default="../etl/data/team_covariates.csv",
        help="Path to team covariates CSV",
    )
    args = parser.parse_args()

    # load data
    data = load_data(args.games, args.covariates)

    # build model
    print("\nBuilding PyMC model...")
    model = build_model(data)
    print(model)

    # fit
    print(f"\nSampling {args.samples} draws × {args.chains} chains...")
    trace = fit_model(model, samples=args.samples, chains=args.chains)

    # summarize
    results = summarize_results(trace, data)

    # save posterior samples for simulator
    np.savez(
        "posterior_samples.npz",
        lam_samples=results["lam_samples"],
        beta_samples=results["beta_samples"],
        tau_samples=results["tau_samples"],
        teams=data["teams"],
    )
    print("\nSaved posterior samples to posterior_samples.npz")


if __name__ == "__main__":
    main()
