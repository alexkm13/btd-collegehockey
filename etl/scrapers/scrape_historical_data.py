"""
Historical data scraper for BTD validation.

Scrapes College Hockey News (team stats + goalie stats) and ESPN (game results)
for multiple seasons.

Usage:
    python scrape_historical.py

Output:
    data/chn_{season}.csv        -- team covariates (FF%_close, PP%)
    data/espn_{season}.csv       -- game results
    data/covariates_all.csv      -- stacked multi-season covariates for stickiness test

Requirements:
    pip install requests beautifulsoup4 pandas
"""

import requests
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEASONS = {
    "20222023": "2022-23",
    "20232024": "2023-24",
    "20242025": "2024-25",
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

DELAY = 2  # seconds between requests


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fetch_soup(url: str) -> BeautifulSoup:
    print(f"  GET {url}")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_table_by_id(soup: BeautifulSoup, table_id: str, min_cols: int = 3) -> list[list[str]]:
    table = soup.find("table", {"id": table_id})
    if table is None:
        print(f"  WARNING: table id='{table_id}' not found")
        return []
    rows = []
    for row in table.find_all("tr"):
        # Skip header rows (use <th>) and section label rows (single <td> spanning table)
        cells = [c.get_text().strip() for c in row.find_all("td")]
        if len(cells) >= min_cols:
            rows.append(cells)
    return rows


def parse_table_by_class(soup: BeautifulSoup, table_class: str, min_cols: int = 3) -> list[list[str]]:
    table = soup.find("table", {"class": table_class})
    if table is None:
        print(f"  WARNING: table class='{table_class}' not found")
        return []
    rows = []
    for row in table.find_all("tr"):
        cells = [c.get_text().strip() for c in row.find_all("td")]
        if len(cells) >= min_cols:
            rows.append(cells)
    return rows


# ---------------------------------------------------------------------------
# CHN: Team Stats Page (standard + advanced + additional tables)
# ---------------------------------------------------------------------------

def scrape_chn_team_stats(season_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrape the main team stats page for a given season.
    URL: https://www.collegehockeynews.com/stats/?season=XXXXXXXX

    Returns (df_standard, df_advanced)
    """
    url = f"https://www.collegehockeynews.com/stats/?season={season_key}"
    soup = fetch_soup(url)

    # --- Standard table (has PP%) ---
    std_rows = parse_table_by_id(soup, "standard")
    df_standard = pd.DataFrame()
    if std_rows:
        std_cols = [
            "Rk", "Team", "GP",
            "G", "GA", "Sh", "Sh%", "ShA", "SV%",
            "PP%", "PK%", "SHG", "SHGA", "FO%", "PIM",
            "G/G", "GA/G", "S/G", "SA/G", "PIM/G",
            "Age", "Ht", "Wt",
        ]
        if len(std_rows[0]) == len(std_cols):
            df_standard = pd.DataFrame(std_rows, columns=std_cols)
            df_standard = df_standard.drop(columns=["Rk", "Age", "Ht", "Wt"])
            numeric = [c for c in df_standard.columns if c != "Team"]
            df_standard[numeric] = df_standard[numeric].apply(pd.to_numeric, errors="coerce")
            print(f"  Standard table: {len(df_standard)} teams")
        else:
            print(f"  WARNING: Standard table has {len(std_rows[0])} cols, expected {len(std_cols)}")
            print(f"  First row: {std_rows[0]}")

    # --- Advanced table (has FF% close) ---
    adv_rows = parse_table_by_id(soup, "advanced")
    df_advanced = pd.DataFrame()
    if adv_rows:
        adv_cols = [
            "Rk", "Team", "GP",
            "SAT", "SATA", "CF%",
            "FF", "FFA", "FF%",
            "ES_SAT", "ES_SATA", "ES_CF%",
            "ES_FF", "ES_FFA", "ES_FF%",
            "PP_SAT", "PP_SATA", "PP_CF%",
            "PP_FF", "PP_FFA", "PP_FF%",
            "Close_SAT", "Close_SATA", "Close_CF%",
            "Close_FF", "Close_FFA", "Close_FF%",
        ]
        if len(adv_rows[0]) == len(adv_cols):
            df_advanced = pd.DataFrame(adv_rows, columns=adv_cols)
            df_advanced = df_advanced.drop(columns=["Rk"])
            numeric = [c for c in df_advanced.columns if c != "Team"]
            df_advanced[numeric] = df_advanced[numeric].apply(pd.to_numeric, errors="coerce")
            print(f"  Advanced table: {len(df_advanced)} teams")
        else:
            print(f"  WARNING: Advanced table has {len(adv_rows[0])} cols, expected {len(adv_cols)}")
            print(f"  First row: {adv_rows[0]}")

    return df_standard, df_advanced


# ---------------------------------------------------------------------------
# CHN: Combine into single season covariate file
# ---------------------------------------------------------------------------

def build_season_covariates(season_key: str) -> pd.DataFrame:
    """
    Merge all CHN tables into one row per team:
    Team, FF%_close, PP%, season
    """
    print(f"\n{'='*60}")
    print(f"CHN: {SEASONS[season_key]}")
    print(f"{'='*60}")

    df_standard, df_advanced = scrape_chn_team_stats(season_key)
    time.sleep(DELAY)

    if df_advanced.empty:
        print(f"  SKIP: No advanced data")
        return pd.DataFrame()

    # Start with FF% close
    result = df_advanced[["Team", "Close_FF%"]].copy()
    result = result.rename(columns={"Close_FF%": "FF%_close"})

    # Merge PP%
    if not df_standard.empty and "PP%" in df_standard.columns:
        result = result.merge(df_standard[["Team", "PP%"]], on="Team", how="left")
    else:
        print(f"  WARNING: PP% not available")
        result["PP%"] = float("nan")

    result["season"] = SEASONS[season_key]
    print(f"  Final: {len(result)} teams, columns: {list(result.columns)}")

    # Sanity check
    print(f"  FF%_close range: {result['FF%_close'].min():.1f} - {result['FF%_close'].max():.1f}")
    if not result["PP%"].isna().all():
        print(f"  PP% range:       {result['PP%'].min():.1f} - {result['PP%'].max():.1f}")

    return result


# ---------------------------------------------------------------------------
# ESPN: Game Results
# ---------------------------------------------------------------------------

def scrape_espn_season(season_key: str) -> pd.DataFrame:
    """Scrape all game results from ESPN's hidden API for a given season."""
    season_start_dates = {
        "20222023": "20221001",
        "20232024": "20231001",
        "20242025": "20241001",
    }

    start_date = season_start_dates[season_key]
    base_url = "https://site.api.espn.com/apis/site/v2/sports/hockey/mens-college-hockey/scoreboard"

    print(f"\n{'='*60}")
    print(f"ESPN: {SEASONS[season_key]}")
    print(f"{'='*60}")

    # Get the season calendar
    cal_url = f"{base_url}?dates={start_date}"
    print(f"  Fetching calendar: {cal_url}")
    resp = requests.get(cal_url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    # Extract all dates
    dates = []
    for entry in data.get("leagues", [{}])[0].get("calendar", []):
        if isinstance(entry, str):
            dates.append(entry[:10].replace("-", ""))
        elif isinstance(entry, dict):
            for sub in entry.get("entries", []):
                start = sub.get("startDate", "")
                if start:
                    dates.append(start[:10].replace("-", ""))

    if not dates:
        print(f"  No calendar found, generating date range")
        year = int(season_key[:4])
        d = datetime(year, 10, 1)
        end = datetime(year + 1, 4, 15)
        while d <= end:
            dates.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)

    dates = sorted(set(dates))
    print(f"  {len(dates)} dates to check")

    # Fetch each date
    all_games = []
    for i, date_str in enumerate(dates):
        url = f"{base_url}?dates={date_str}"
        try:
            resp = requests.get(url, headers=HEADERS)
            resp.raise_for_status()
            day_data = resp.json()
        except Exception as e:
            print(f"  Error on {date_str}: {e}")
            continue

        events = day_data.get("events", [])
        for event in events:
            game = parse_espn_event(event)
            if game:
                all_games.append(game)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(dates)} dates, {len(all_games)} games so far")

        time.sleep(0.3)

    print(f"  Total games: {len(all_games)}")
    return pd.DataFrame(all_games)


def classify_outcome(home_score: int, away_score: int, period: int) -> tuple[str, str]:
    """Classify game outcome. Returns (home_outcome, away_outcome)."""
    if period == 3:  # regulation
        if home_score > away_score:
            return "RW", "RL"
        else:
            return "RL", "RW"
    else:  # overtime
        if home_score > away_score:
            return "OW", "OL"
        elif away_score > home_score:
            return "OL", "OW"
        else:
            return "OW", "OL"  # caller handles winner flag


def parse_espn_event(event: dict) -> dict | None:
    """Parse a single ESPN event into a game result dict."""
    comp = event["competitions"][0]
    status = comp["status"]

    if not status["type"]["completed"]:
        return None

    period = status["period"]
    detail = status["type"].get("detail", "")

    home = comp["competitors"][0]
    away = comp["competitors"][1]

    home_team = home["team"]["displayName"]
    away_team = away["team"]["displayName"]
    home_score = int(home["score"])
    away_score = int(away["score"])

    # Handle ties / shootouts
    if home_score == away_score and period >= 4:
        if home.get("winner", False):
            home_outcome, away_outcome = "OW", "OL"
        elif away.get("winner", False):
            home_outcome, away_outcome = "OL", "OW"
        else:
            home_outcome, away_outcome = "T", "T"
    else:
        home_outcome, away_outcome = classify_outcome(home_score, away_score, period)

    return {
        "date": event["date"][:10],
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "period": period,
        "detail": detail,
        "home_outcome": home_outcome,
        "away_outcome": away_outcome,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- CHN covariates ----
    all_covariates = []
    for season_key in SEASONS:
        df = build_season_covariates(season_key)
        if not df.empty:
            outpath = DATA_DIR / f"chn_{SEASONS[season_key]}.csv"
            df.to_csv(outpath, index=False)
            print(f"  Saved: {outpath}")
            all_covariates.append(df)

    if all_covariates:
        df_all = pd.concat(all_covariates, ignore_index=True)
        df_all.to_csv(DATA_DIR / "covariates_all.csv", index=False)
        print(f"\nStacked covariates: {len(df_all)} rows -> data/covariates_all.csv")

    # ---- ESPN game results ----
    for season_key in SEASONS:
        df = scrape_espn_season(season_key)
        if not df.empty:
            outpath = DATA_DIR / f"espn_{SEASONS[season_key]}.csv"
            df.to_csv(outpath, index=False)
            print(f"  Saved: {outpath}")

    print(f"\n{'='*60}")
    print("DONE. Next: python stickiness.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()