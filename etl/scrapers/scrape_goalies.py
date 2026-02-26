import requests
from bs4 import BeautifulSoup
import pandas as pd


URL = "https://www.collegehockeynews.com/stats/overall-goalie.php"


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.text, "html.parser")


def parse_table(soup: BeautifulSoup, table_class: str) -> list[list[str]]:
    table = soup.find("table", {"class": table_class})
    rows = []
    for row in table.find_all("tr"):
        cells = [c.get_text().strip() for c in row.find_all("td")]
        if cells:
            rows.append(cells)
    return rows


def build_df_goalies(rows: list[list[str]]) -> pd.DataFrame:
    cols = [
        "Rk",
        "Name",
        "Team",
        "Yr",
        "GP",
        "W",
        "L",
        "T",
        "GA",
        "MIN",
        "GAA",
        "SH",
        "SV",
        "SV%",
        "CHIP",
        "xGa+",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop(columns=["Rk"])
    df["Name"] = df["Name"].str.replace("\xa0", " ")
    df["xGa+"] = df["xGa+"].str.replace("+", "")

    numeric = [c for c in df.columns if c not in ("Team",)]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    return df


def main():
    soup = fetch_soup(URL)

    rows = parse_table(soup, "data sortable")
    df_goalies = build_df_goalies(rows)

    df_goalies = (
        df_goalies.groupby("Team").agg({"xGa+": "sum", "MIN": "sum"}).reset_index()
    )
    df_goalies["GSAx_60"] = df_goalies["xGa+"] * 60 / df_goalies["MIN"]
    df_goalies.to_csv("chn_goalie_stats.csv", index=False)


if __name__ == "__main__":
    main()
