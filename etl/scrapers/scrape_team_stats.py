import requests
from bs4 import BeautifulSoup
import pandas as pd


URL = "https://www.collegehockeynews.com/stats/#adv"


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.text, "html.parser")


def parse_table(soup: BeautifulSoup, table_id: str) -> list[list[str]]:
    table = soup.find("table", {"id": table_id})
    rows = []
    for row in table.find_all("tr"):
        cells = [c.get_text().strip() for c in row.find_all("td")]
        if cells:
            rows.append(cells)
    return rows

def build_standard_df(rows: list[list[str]]) -> pd.DataFrame:
    cols = [
        "Rk", "Team", "GP",
        "G", "GA", "Sh", "Sh%", "ShA", "SV%",
        "PP%", "PK%", "SHG", "SHGA", "FO%", "PIM",
        "G/G", "GA/G", "S/G", "SA/G", "PIM/G",
        "Age", "Ht", "Wt",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop(columns=["Rk", "Age", "Ht", "Wt"])

    numeric = [c for c in df.columns if c not in ("Team",)]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    return df


def build_advanced_df(rows: list[list[str]]) -> pd.DataFrame:
    cols = [
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
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop(columns=["Rk"])

    numeric = [c for c in df.columns if c not in ("Team",)]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    return df


def build_additional_df(rows: list[list[str]]) -> pd.DataFrame:
    """Pull Team + ES_GF% by index position from the Additional table."""
    data_rows = [r for r in rows if len(r) == max(len(r) for r in rows)]
    df = pd.DataFrame(data_rows)
    df = df[[1, 8]]  # Team and ES_GF%
    df.columns = ["Team", "ES_GF%"]
    df["ES_GF%"] = pd.to_numeric(df["ES_GF%"], errors="coerce")
    return df


def main():
    soup = fetch_soup(URL)

    std_rows = parse_table(soup, "standard")
    adv_rows = parse_table(soup, "advanced")
    add_rows = parse_table(soup, "additional")

    df_standard = build_standard_df(std_rows)
    df_advanced = build_advanced_df(adv_rows)
    df_additional = build_additional_df(add_rows)

    df = df_advanced.merge(df_additional[["Team", "ES_GF%"]], on="Team")
    df = df.merge(df_standard[["Team", "PP%"]], on="Team")
    df.to_csv("chn_team_stats.csv", index=False)


if __name__ == "__main__":
    main()