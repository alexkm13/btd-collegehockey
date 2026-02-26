import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import warnings
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

OUTCOME_MAP = {"RW": 0, "OW": 1, "T": 2, "OL": 3, "RL": 4}
P_COEFS = np.array([1.0, 2 / 3, 1 / 2, 1 / 3, 0.0])
O_COEFS = np.array([0.0, 1.0, 1.0, 1.0, 0.0])

ESPN_TO_CHN = {
    "Air Force Falcons": "Air Force",
    "Alaska Anchorage Seawolves": "Alaska-Anchorage",
    "Alaska Bulldogs": "Alaska",
    "Arizona State Sun Devils": "Arizona State",
    "Army Black Knights": "Army",
    "Augustana University (SD) Vikings": "Augustana",
    "Bemidji State Beavers": "Bemidji State",
    "Bentley Falcons": "Bentley",
    "Boston College Eagles": "Boston College",
    "Boston University Terriers": "Boston University",
    "Bowling Green Falcons": "Bowling Green",
    "Brown Bears": "Brown",
    "Canisius Golden Griffins": "Canisius",
    "Clarkson Golden Knights": "Clarkson",
    "Colgate Raiders": "Colgate",
    "Colorado College Tigers": "Colorado College",
    "Cornell Big Red": "Cornell",
    "Dartmouth Big Green": "Dartmouth",
    "Denver Pioneers": "Denver",
    "Ferris State Bulldogs": "Ferris State",
    "Harvard Crimson": "Harvard",
    "Holy Cross Crusaders": "Holy Cross",
    "Lake Superior State Lakers": "Lake Superior",
    "Lindenwood Lions": "Lindenwood",
    "Long Island University Long Island University": "Long Island",
    "Maine Black Bears": "Maine",
    "Massachusetts Minutemen": "Massachusetts",
    "Mercyhurst Lakers": "Mercyhurst",
    "Merrimack Warriors": "Merrimack",
    "Miami (OH) RedHawks": "Miami",
    "Michigan State Spartans": "Michigan State",
    "Michigan Tech Huskies": "Michigan Tech",
    "Michigan Wolverines": "Michigan",
    "Minnesota Duluth Bulldogs": "Minnesota-Duluth",
    "Minnesota Golden Gophers": "Minnesota",
    "Minnesota State Mavericks": "Minnesota State",
    "New Hampshire Wildcats": "New Hampshire",
    "Niagara Purple Eagles": "Niagara",
    "North Dakota Fighting Hawks": "North Dakota",
    "Northeastern Huskies": "Northeastern",
    "Northern Michigan Wildcats": "Northern Michigan",
    "Notre Dame Fighting Irish": "Notre Dame",
    "Ohio State Buckeyes": "Ohio State",
    "Omaha Mavericks": "Omaha",
    "Penn State Nittany Lions": "Penn State",
    "Princeton Tigers": "Princeton",
    "Providence Friars": "Providence",
    "Quinnipiac Bobcats": "Quinnipiac",
    "RIT Tigers": "RIT",
    "Rensselaer Engineers": "RPI",
    "Robert Morris Colonials": "Robert Morris",
    "Sacred Heart Pioneers": "Sacred Heart",
    "St. Cloud State Huskies": "St. Cloud State",
    "St. Lawrence Saints": "St. Lawrence",
    "St. Thomas - Minnesota Tommies": "St. Thomas",
    "Stonehill Stonehill": "Stonehill",
    "UConn Huskies": "Connecticut",
    "UMass Lowell  River Hawks": "Mass.-Lowell",
    "Union Dutchmen": "Union",
    "Vermont Catamounts": "Vermont",
    "Western Michigan Broncos": "Western Michigan",
    "Wisconsin Badgers": "Wisconsin",
    "Yale Bulldogs": "Yale",
}


def load_season(season_label: str):
    """Load CHN covariates and ESPN games for a given season."""
    chn = pd.read_csv(f"data/chn_{season_label}.csv")
    espn = pd.read_csv(f"data/espn_{season_label}.csv")

    # map ESPN names → CHN names
    espn["home_team"] = espn["home_team"].map(ESPN_TO_CHN)
    espn["away_team"] = espn["away_team"].map(ESPN_TO_CHN)

    # drop non-D1 or unmapped teams
    espn = espn.dropna(subset=["home_team", "away_team"])

    # drop bad records (period=0)
    espn = espn[espn["period"] > 0]

    print(f"  {season_label}: {len(espn)} games, {len(chn)} teams with covariates")

    return chn, espn


def prepare_model_data(
    games: pd.DataFrame, covariates: pd.DataFrame, cov_cols: list[str]
):
    """
    Build arrays for PyMC model from games and covariates.
    Only includes games where both teams have covariates.
    """
    cov_teams = set(covariates["Team"])
    mask = games["home_team"].isin(cov_teams) & games["away_team"].isin(cov_teams)
    games = games[mask].copy()

    # build team index from covariates
    teams = sorted(covariates["Team"].unique())
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = games["home_team"].map(team_to_idx).values.astype(int)
    away_idx = games["away_team"].map(team_to_idx).values.astype(int)
    outcome_idx = games["home_outcome"].map(OUTCOME_MAP).values.astype(int)

    # build covariate matrix
    if cov_cols:
        X_raw = covariates.set_index("Team").loc[teams, cov_cols].values.astype(float)
        X_mean = X_raw.mean(axis=0)
        X_std = X_raw.std(axis=0)
        X_std[X_std == 0] = 1
        X = (X_raw - X_mean) / X_std
    else:
        X = None

    return {
        "home_idx": home_idx,
        "away_idx": away_idx,
        "outcome_idx": outcome_idx,
        "X": X,
        "n_teams": n_teams,
        "n_games": len(games),
        "teams": teams,
        "team_to_idx": team_to_idx,
        "n_covariates": len(cov_cols),
    }


def build_btd_model(data: dict) -> pm.Model:
    """Build BTD model. If n_covariates=0, this is the base Whelan model."""
    n_teams = data["n_teams"]
    n_cov = data["n_covariates"]
    home_idx = data["home_idx"]
    away_idx = data["away_idx"]
    outcome_idx = data["outcome_idx"]

    with pm.Model() as model:
        alpha_free = pm.Normal("alpha_free", mu=0, sigma=1, shape=n_teams - 1)
        alpha = pt.concatenate([alpha_free, -pt.sum(alpha_free, keepdims=True)])

        tau = pm.Normal("tau", mu=0, sigma=1)

        if n_cov > 0:
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_cov)
            X_tensor = pt.as_tensor_variable(data["X"])
            lam = alpha + pt.dot(X_tensor, beta)
        else:
            lam = alpha

        pm.Deterministic("lambda", lam)

        gamma = lam[home_idx] - lam[away_idx]

        p_coefs = pt.as_tensor_variable(P_COEFS)
        o_coefs = pt.as_tensor_variable(O_COEFS)
        scores = gamma[:, None] * p_coefs[None, :] + tau * o_coefs[None, :]

        pm.Categorical("obs", p=pm.math.softmax(scores, axis=1), observed=outcome_idx)

    return model


def fit_btd(
    data: dict, samples: int = 1000, chains: int = 4
) -> pm.backends.base.MultiTrace:
    """Fit the BTD model and return the trace."""
    model = build_btd_model(data)
    with model:
        trace = pm.sample(
            draws=samples,
            tune=500,
            chains=chains,
            cores=4,
            target_accept=0.9,
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
        )
    return trace


def predict_game_probs(
    trace, train_data: dict, home_team: str, away_team: str
) -> np.ndarray:
    """Predict P(home wins) for a single game using posterior samples."""
    team_to_idx = train_data["team_to_idx"]

    # if a team wasn't in the training data, we can't predict
    if home_team not in team_to_idx or away_team not in team_to_idx:
        return None

    hi = team_to_idx[home_team]
    ai = team_to_idx[away_team]

    # extract posterior samples
    lam_samples = trace.posterior["lambda"].values.reshape(-1, train_data["n_teams"])
    tau_samples = trace.posterior["tau"].values.flatten()

    n_draws = len(tau_samples)
    probs_home_win = np.zeros(n_draws)

    for d in range(n_draws):
        gamma = lam_samples[d, hi] - lam_samples[d, ai]
        tau = tau_samples[d]

        # compute softmax over 5 outcomes
        scores = P_COEFS * gamma + O_COEFS * tau
        exp_scores = np.exp(scores - scores.max())  # numerical stability
        probs = exp_scores / exp_scores.sum()

        # P(home wins) = P(RW) + P(OW)
        probs_home_win[d] = probs[0] + probs[1]

    return probs_home_win.mean()


def compute_brier_score(predictions: list[tuple[float, int]]) -> float:
    """
    Compute Brier score from list of (predicted_prob, actual_outcome) tuples.
    actual_outcome: 1 if home won (RW or OW), 0 otherwise.
    """
    if not predictions:
        return float("nan")

    total = 0.0
    for p_hat, y in predictions:
        total += (p_hat - y) ** 2

    return total / len(predictions)


def evaluate_model(
    trace,
    train_data: dict,
    test_games: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Evaluate a trained model on test games.
    Returns Brier score and prediction details.
    """
    predictions = []
    skipped = 0

    for _, game in test_games.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        p_home_win = predict_game_probs(trace, train_data, home, away)
        if p_home_win is None:
            skipped += 1
            continue

        # actual outcome-- 1 if home won (RW or OW), 0 otherwise
        actual = 1 if game["home_outcome"] in ("RW", "OW") else 0
        predictions.append((p_home_win, actual))

    brier = compute_brier_score(predictions)
    n_pred = len(predictions)

    print(
        f"  {model_name}: Brier = {brier:.4f} ({n_pred} games predicted, {skipped} skipped)"
    )

    sq_errors = [(p - a)**2 for p, a in predictions]
    
    return {
        "model": model_name,
        "brier_score": brier,
        "n_predictions": n_pred,
        "n_skipped": skipped,
        "sq_errors": sq_errors,
    }


def run_validation_pair(train_season: str, test_season: str) -> list[dict]:
    """
    Train on season N, predict season N+1.
    Returns list of result dicts (one per model).
    """
    print(f"\n{'=' * 60}")
    print(f"VALIDATION: Train on {train_season} → Predict {test_season}")
    print(f"{'=' * 60}")

    # Load data
    chn_train, espn_train = load_season(train_season)
    chn_test, espn_test = load_season(test_season)

    results = []

    # base Whelan (no covariates)
    print("\n--- Model A: Base Whelan (no covariates) ---")
    data_a = prepare_model_data(espn_train, chn_train, cov_cols=[])
    print(f"  Training on {data_a['n_games']} games, {data_a['n_teams']} teams")
    trace_a = fit_btd(data_a)
    result_a = evaluate_model(trace_a, data_a, espn_test, "Base Whelan")
    result_a["train_season"] = train_season
    result_a["test_season"] = test_season
    results.append(result_a)

    # sticky covariates (FF%_close + PP%)
    print("\n--- Model B: Sticky covariates (FF%_close + PP%) ---")
    data_b = prepare_model_data(espn_train, chn_train, cov_cols=["FF%_close", "PP%"])
    print(
        f"  Training on {data_b['n_games']} games, {data_b['n_teams']} teams, 2 covariates"
    )
    trace_b = fit_btd(data_b)
    result_b = evaluate_model(trace_b, data_b, espn_test, "Sticky (FF%+PP%)")
    result_b["train_season"] = train_season
    result_b["test_season"] = test_season
    results.append(result_b)

    if len(result_a["sq_errors"]) == len(result_b["sq_errors"]):
        t_stat, p_val = stats.ttest_rel(
            result_a["sq_errors"], 
            result_b["sq_errors"], 
            alternative='greater' # Testing if Base errors are strictly > Covariate errors
        )
        print(f"\n  [Statistical Significance]")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value:     {p_val:.4f}")
        
        # Attach it to the covariate model's result dict for the final dataframe
        result_b["p_value"] = p_val
        result_a["p_value"] = float('nan')
    else:
        print("\n  [Warning] Mismatched prediction counts, skipping t-test.")

    # Remove the heavy arrays before returning to keep the dataframe clean
    del result_a["sq_errors"]
    del result_b["sq_errors"]

    return results

def main():
    # season pairs for cross-season validation
    pairs = [
        ("2022-23", "2023-24"),
        ("2023-24", "2024-25"),
    ]

    all_results = []

    for train_s, test_s in pairs:
        results = run_validation_pair(train_s, test_s)
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    print(f"\n\n{'=' * 60}")
    print("BRIER SCORE SUMMARY")
    print(f"{'=' * 60}")
    print(
        df[
            ["train_season", "test_season", "model", "brier_score", "n_predictions"]
        ].to_string(index=False)
    )

    # average across pairs
    print("\nAverage Brier Score by Model:")
    avg = df.groupby("model")["brier_score"].mean()
    for model, brier in avg.items():
        print(f"  {model:25s}: {brier:.4f}")

    best = avg.idxmin()
    improvement = (avg.max() - avg.min()) / avg.max() * 100
    print(f"\nBest model: {best}")
    print(f"Improvement: {improvement:.1f}% lower Brier score")

    df.to_csv("data/brier_results.csv", index=False)
    print("\nSaved: data/brier_results.csv")


if __name__ == "__main__":
    main()
