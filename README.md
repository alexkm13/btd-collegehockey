# Covariate-Enhanced Bayesian Bradley-Terry-Davidson Model for NCAA D1 Men's Hockey

Predicts conference regular-season winners and NCAA tournament outcomes by extending the [Whelan & Klein (2021)](https://arxiv.org/abs/2112.01267) paired comparison model with team-level performance covariates.

<!-- screenshot of bracket viz goes here -->
<!-- ![Tournament Probabilities](docs/tournament_bracket.png) -->

## What This Does

Whelan & Klein estimate team strength purely from game results such as wins, losses, and overtime outcomes. This project extends their model by asking *why* teams win, not just *that* they win. Three covariates are added to the team strength equation:

| Covariate | What It Measures | $\beta$ (standardized) | P($\beta$ > 0) |
|-----------|-----------------|-------------------|-----------|
| **FF% Close** | 5-on-5 possession in close game situations | +0.619 $\pm$ 0.158 | 100.0% |
| **PP%** | Power play conversion rate | +0.371 $\pm$ 0.163 | 98.9% |
| **GSAx/60** | Goals saved above expected per 60 min (goaltending quality) | +0.300 $\pm$ 0.146 | 98.1% |

All three are statistically significant. The model decomposes team strength into three orthogonal dimensions: how you play at even strength, how you play on special teams, and how your goalie plays.

## How It Works

Each team's strength is modeled as:


$$\lambda_i = \alpha_i + \beta_1 \cdot \text{FF\}_{\text{close}}% + \beta_2 \cdot \text{PP}% + \beta_3 \cdot \text{GSAx/60}$$

where $\alpha_{i}$ is a team specific intercept and the $\beta$ coefficients are shared across all teams. Game outcomes follow a 5 category softmax which incl. regulation win, overtime win, tie, overtime loss, regulation loss, extending Whelan's 4-outcome model to handle NCAA ties.

The posterior is sampled via NUTS (4 chains × 2,000 draws). Team strengths are then fed into a Monte Carlo simulator of 10,000 iterations that plays out full conference round-robins and single-elimination tournament brackets.

## Validation

### Covariate Stickiness

Before using covariates for cross-season prediction, we test whether they capture durable program traits or transient roster quality by computing year-over-year Pearson correlations across all D1 teams:

| Covariate | 2022-23 → 2023-24 | 2023-24 → 2024-25 | Interpretation |
|-----------|:------------------:|:------------------:|----------------|
| **FF% Close** | r = 0.663 | r = 0.554 | Sticky, program-level trait like the coaching system, or recruiting |
| **PP%** | r = 0.414 | r = 0.327 | Moderate, partially system & partially personnel |
| **GSAx/60** | — | — | Not testable (CHN data unavailable for historical seasons) |

Possession quality persists across seasons despite roster turnover. Special teams are moderately durable. Goaltending is hypothesized to be roster-dependent (low stickiness) but cannot be validated with available data.

### Out-of-Sample Brier Scores

We train on season N and predict every game in season N+1, comparing the covariate-enhanced model against a base Whelan model with no covariates:

| Train → Test | Base Whelan | Sticky (FF% + PP%) | $\Delta$ Brier | DM Test | p-value |
|:------------:|:-----------:|:-------------------:|:-------:|:----------------:|:-------:|
| 2022-23 → 2023-24 | 0.2456 | **0.2445** | −0.0010 | 0.906 | 0.1826 |
| 2023-24 → 2024-25 | 0.2319 | **0.2285** | −0.0033 | **2.265** | **0.0118** |

*Note: The Diebold-Mariano test evaluates the significance of the squared prediction errors. The 2024-25 validation demonstrates statistically significant structural alpha* $(t(1093) = 2.265, p < 0.05)$ *despite the severe penalty of cross-season roster turnover.*

**Aggregate Out-of-Sample Performance (2022–2025):**
Across both test seasons (2,076 total out-of-sample predictions), the inclusion of possession and special teams covariates reduced the average Brier score from **0.2387** to **0.2365**. This represents an overall **0.9% reduction** in prediction error against an already optimized Bayesian baseline.

The covariate model outperforms in both season pairs. The improvement is conservative due to full season-to-season prediction with roster churn, making this a very difficult test. Within season validation would show larger gains but requires play-by-play data unavailable at the college level as of right now.

## Results

### 2025–26 Conference Predictions

Run the conference simulator to get regular-season winner probabilities, average point totals, and top-4 likelihoods for all six conferences which are Atlantic Hockey, Big Ten, CCHA, ECAC, Hockey East, NCHC.

### 2025–26 NCAA Tournament Predictions

| Team | Seed | Reg. Final | Frozen Four | Title Game | Champion |
|------|:----:|:----------:|:-----------:|:----------:|:--------:|
| Michigan State | 1 | 85.6% | 62.5% | 41.6% | **28.0%** |
| Michigan | 2 | 73.0% | 45.1% | 22.4% | **13.0%** |
| North Dakota | 3 | 68.0% | 42.6% | 23.6% | **11.8%** |
| Western Michigan | 4 | 60.5% | 32.7% | 17.9% | 8.6% |
| Penn State | 5 | 60.9% | 33.4% | 19.0% | 8.6% |
| Quinnipiac | 7 | 53.4% | 24.4% | 9.7% | 4.7% |
| Minnesota Duluth | 8 | 58.3% | 20.0% | 9.4% | 4.6% |
| Providence | 6 | 57.0% | 26.0% | 11.6% | 4.4% |
| Denver | 9 | 46.6% | 20.0% | 7.4% | 3.0% |
| Boston College | 12 | 39.1% | 17.0% | 7.3% | 2.8% |
| Wisconsin | 13 | 39.5% | 16.9% | 7.6% | 2.8% |
| Dartmouth | 11 | 43.0% | 16.8% | 7.2% | 2.5% |
| Cornell | 10 | 41.7% | 12.5% | 4.9% | 2.1% |
| Connecticut | 14 | 32.0% | 14.6% | 5.8% | 2.0% |
| St. Thomas | 15 | 27.0% | 10.5% | 3.2% | 1.0% |
| Bentley | 16 | 14.4% | 5.1% | 1.5% | 0.3% |


## Data Pipeline

**Extract:** Game results from the ESPN API (961 games, 2025–26 season). Team statistics from College Hockey News (63 teams, advanced + standard + goalie tables). Historical data for 2022–23, 2023–24, and 2024–25 seasons for validation.

**Transform:** Map ESPN team names to CHN names. Filter bad records like 24 games with missing scores. Classify outcomes (RW/OW/T/OL/RL) by period count. Aggregate goalie stats to team-level GSAx/60 (weighted by minutes, 33% minimum threshold). Standardize covariates to zero mean, unit variance.

**Load:** Two clean CSVs — `game_results.csv` (one row per game) and `team_covariates.csv` (one row per team).

## Quickstart

```bash
# Install dependencies
pip install pymc arviz numpy pandas requests beautifulsoup4

# Scrape current season data
cd etl/scrapers
python scrape_espn_games.py
python scrape_team_stats.py

# Merge into model inputs
cd ..
python merge.py

# Fit the model (~12 seconds on M4-series Mac)
cd ../engine
python btd.py

# Simulate conferences
python simulator.py --sims 10000

# Simulate the NCAA tournament
python simulator.py --tournament --sims 10000

# Run validation (scrapes historical data, ~5 min)
cd ../validation
python ../etl/scrapers/scrape_historical.py
python stickiness.py
python brier_validation.py
```

## Model Diagnostics

| Metric | Value | Target |
|--------|-------|--------|
| Max $\hat{R}$ | 1.0000 | $\approx$ 1.0 |
| Min ESS | 1,886 | $\gt$ 400 |
| Divergences | 0 | 0 |

## Extensions from Whelan & Klein (2021)

1. **Covariate extension** — Team strength is partially explained by observable metrics rather than being purely latent. This allows the model to identify teams winning unsustainably (lucky) or losing despite strong process (unlucky).

2. **Ties as a fifth outcome** — NCAA conference games can end in a tie. The tie outcome extends Whelan's 4-outcome softmax to a 5-category likelihood. For a game between teams *i* and *j*, the probability of outcome *k* is:

$$P(\text{outcome } k \mid i, j) = \frac{\exp(s_k)}{\sum_{K} \exp(s_K)}$$

where the linear predictor for outcome $k$ is defined as: $$s_k = p_k \cdot \gamma_{ij} + o_k \cdot \tau$$

and the strength difference between the two teams is: $$\gamma_{ij} = \lambda_i - \lambda_j$$.
 
The outcome parameters ($p_k$, $o_k$) encode the information content of each result:

| Outcome | $p_k$ | $o_k$ | Interpretation |
|---------|:--:|:--:|----------------|
| Regulation Win | 1 | 0 | Full win, no overtime (strongest signal) |
| Overtime Win | 2/3 | 1 | Partial win, required extra time |
| Tie | 1/2 | 1 | Draw, consistent with Davidson (1970) |
| Overtime Loss | 1/3 | 1 | Partial loss, required extra time |
| Regulation Loss | 0 | 0 | Full loss (strongest signal) |

The $\tau$ parameter controls the overall frequency of overtime outcomes. Posterior estimate: $\tau$ = −1.760 $\pm$ 0.085, meaning overtime results are rare relative to regulation outcomes (exp(−1.76) ≈ 0.17).

3. **Monte Carlo simulation layer** — Posterior draws feed directly into a season and tournament simulator, propagating parameter uncertainty through to the final predictions.

4. **Cross-season validation** — Covariate stickiness analysis and out-of-sample Brier scores demonstrate that process metrics capture durable program level traits that improve prediction across seasons despite roster turnover.

## Limitations

- Covariates are season-level aggregates, not game-level. Within-season rolling updates would improve the model but require play-by-play data unavailable at the college level.
- GSAx/60 cannot be validated cross-season due to CHN data limitations for historical goalie stats. It is hypothesized to have low year-over-year stickiness as the stat is heavily goalie-dependent, not program-dependent.
- No home-ice advantage term. Could be added as a contest-level covariate.
- Covariate selection was theory-driven using three orthogonal dimensions of team quality, not data-driven search. Additional covariates i.e. PK%, score-adjusted metrics could be tested.
- Cross-season validation understates the model's predictive power due to roster turnover. The Brier score improvement would likely be larger with within-season point-in-time covariates.

## References

- Whelan, J.T. & Klein, M.R. (2021). [*Bayesian Bradley-Terry-Davidson Analysis of Multiple-Outcome Competitions.*](https://arxiv.org/abs/2112.01267) arXiv:2112.01267.
- Bradley, R.A. & Terry, M.E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345.
- Davidson, R.R. (1970). On extending the Bradley-Terry model to accommodate ties in paired comparison experiments. *JASA*, 65(329), 317–328.
- Dittrich, R., Hatzinger, R. & Katzenbeisser, W. (1998). Subject-specific covariates in the Bradley-Terry model. *Computational Statistics*, 13, 295–308.
- Turner, H. & Firth, D. (2012). Bradley-Terry models in R: The BradleyTerry2 package. *Journal of Statistical Software*, 48(9), 1–21.

## License

MIT License
