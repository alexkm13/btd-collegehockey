import requests
import pandas as pd
import time

BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/hockey/mens-college-hockey/scoreboard"

def get_calendar() -> list[str]:
    """Fetch the season calendar dates from ESPN API."""
    resp = requests.get(BASE_URL)
    data = resp.json()
    # calendar entries are ISO timestamps, get YYYYMMDD
    dates = []
    for ts in data["leagues"][0]["calendar"]:
        date_str = ts[:10].replace("-", "") 
        dates.append(date_str)
    return dates


def classify_outcome(home_score: int, away_score: int, period: int) -> tuple[str, str]:
    """
    Classify game outcome for BTD model.
    Returns (home_outcome, away_outcome) from {RW, OW, OL, RL}.
    """
    if period == 3:  # regulation
        if home_score > away_score:
            return "RW", "RL"
        else:
            return "RL", "RW"
    else:  # overtime (period >= 4)
        if home_score > away_score:
            return "OW", "OL"
        elif away_score > home_score:
            return "OL", "OW"
        else:
            # shootout: scores tied, check winner flag
            # ESPN marks winner=true on the SO winner
            return "OW", "OL"  # caller handles winner flag


def parse_game(event: dict) -> dict | None:
    """Parse a single game event into a row for the DataFrame."""
    comp = event["competitions"][0]
    status = comp["status"]

    # skip games that aren't finished
    if not status["type"]["completed"]:
        return None

    period = status["period"]
    detail = status["type"].get("detail", "")

    # home is 0, away is 1
    home = comp["competitors"][0]
    away = comp["competitors"][1]

    home_team = home["team"]["displayName"]
    away_team = away["team"]["displayName"]
    home_score = int(home["score"])
    away_score = int(away["score"])

    # shootouts
    if home_score == away_score and period >= 4:
        # winner flag
        if home.get("winner", False):
            home_outcome, away_outcome = "OW", "OL"
        elif away.get("winner", False):
            home_outcome, away_outcome = "OL", "OW"
        else:
            # actual tie
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


def scrape_all_games() -> pd.DataFrame:
    """Scrape all games from the season."""
    dates = get_calendar()
    print(f"Found {len(dates)} game dates in calendar")

    all_games = []

    for i, date in enumerate(dates):
        resp = requests.get(BASE_URL, params={"dates": date})
        data = resp.json()

        for event in data.get("events", []):
            game = parse_game(event)
            if game:
                all_games.append(game)

        if (i + 1) % 10 == 0:
            print(f"  Scraped {i + 1}/{len(dates)} dates ({len(all_games)} games so far)")

        time.sleep(0.3)  # give ESPN a lil break

    print(f"Done: {len(all_games)} total games")
    return pd.DataFrame(all_games)


def main():
    df = scrape_all_games()

    # Print summary
    print(f"\nOutcome distribution:")
    print(df["home_outcome"].value_counts())

    print(f"\nSample games:")
    print(df.head(10).to_string(index=False))

    df.to_csv("espn_game_results.csv", index=False)
    print(f"\nSaved to espn_game_results.csv")


if __name__ == "__main__":
    main()