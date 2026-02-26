import pandas as pd

df_team = pd.read_csv("data/chn_team_stats.csv")
df_goalie = pd.read_csv("data/chn_goalie_stats.csv")

df = df_team.merge(df_goalie[["Team", "GSAx_60"]], on="Team")

df.to_csv("chn_combined.csv", index=False)
