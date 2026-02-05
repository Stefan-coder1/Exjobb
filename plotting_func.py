import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch, VerticalPitch  
from pathlib import Path
import os
import json
import csv
from tqdm import tqdm
import logging
from dotenv import load_dotenv
from pathlib import Path
import os
from json import JSONDecodeError
from matplotlib.patches import Polygon, Rectangle

def build_curve_fixed(df, match_id, team_id, max_poss_minutes=9.0):
    max_poss_seconds = max_poss_minutes * 60

    sub = df[(df["match_id"] == match_id)].copy()

    # ensure possession_team_id exists (same as your calc_possession_time)
    if "possession_team_id" not in sub.columns:
        if "possession_team" in sub.columns:
            sub["possession_team_id"] = sub["possession_team"].apply(
                lambda x: x.get("id") if isinstance(x, dict) else np.nan
            )
        else:
            raise ValueError("Need possession_team_id (or possession_team dict) in df.")

    # keep only possessions owned by this team
    sub = sub[sub["possession_team_id"] == team_id].copy()

    # time base (ms) + duration, same as your tempo possession calc
    sub["t"] = pd.to_timedelta(sub["timestamp"])
    if "duration" not in sub.columns:
        sub["duration"] = 0.0
    sub["duration"] = pd.to_numeric(sub["duration"], errors="coerce").fillna(0.0)
    sub["t_end"] = sub["t"] + pd.to_timedelta(sub["duration"], unit="s")

    # 1) possession segments (start/end per possession)
    seg = (sub.groupby(["possession"], as_index=False)
             .agg(start=("t", "min"), end=("t_end", "max")))
    seg["possession_seconds"] = (seg["end"] - seg["start"]).dt.total_seconds().clip(lower=0)

    # 2) passes per possession (only within those team-owned possessions)
    passes = sub[sub["type"] == "Pass"]
    pass_counts = (passes.groupby("possession").size()
                         .rename("n_passes")
                         .reset_index())

    seg = seg.merge(pass_counts, on="possession", how="left")
    seg["n_passes"] = seg["n_passes"].fillna(0).astype(int)

    # 3) order possessions chronologically, then accumulate time+passes together
    seg = seg.sort_values("start").reset_index(drop=True)
    seg["cum_poss_time"] = seg["possession_seconds"].cumsum()
    seg["cum_passes"] = seg["n_passes"].cumsum()

    # 4) truncate to fixed possession window
    seg = seg[seg["cum_poss_time"] <= max_poss_seconds].copy()

    # return x,y for plotting (end-of-possession points)
    return seg["cum_poss_time"].to_numpy(), seg["cum_passes"].to_numpy()




def plot_two_matches(df, a, b):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)

    for ax, row, title in zip(
        axes,
        [a, b],
        ["Most tempo", "Least tempo"]
    ):
        pitch.draw(ax=ax)

        subset = df[
            (df["match_id"] == row["match_id"]) &
            (df["team_id"] == row["team_id"]) 
            #&(df["is_progressive"])
        ]

        passes  = subset[subset["type"] == "Pass"]
        carries = subset[subset["type"] == "Carry"]
        dribbles = subset[subset["type"] == "Dribble"]

        # --- passes (blue) ---
        pitch.lines(
            passes["x"], passes["y"],
            passes["endx"], passes["endy"],
            ax=ax,
            lw=2, alpha=0.7, color="tab:blue", label="Pass"
        )
        pitch.scatter(
            passes["x"], passes["y"],
            ax=ax, s=20, color="tab:blue"
        )

        """        # --- carries (orange) ---
        pitch.lines(
            carries["x"], carries["y"],
            carries["endx"], carries["endy"],
            ax=ax,
            lw=2, alpha=0.7, color="tab:orange", label="Carry"
        )
        pitch.scatter(
            carries["x"], carries["y"],
            ax=ax, s=20, color="tab:orange"
        )

        # --- dribbles (green) ---
        pitch.lines(
            dribbles["x"], dribbles["y"],
            dribbles["endx"], dribbles["endy"],
            ax=ax,
            lw=2, alpha=0.7, color="tab:green", label="Dribble"
        )
        pitch.scatter(
            dribbles["x"], dribbles["y"],
            ax=ax, s=20, color="tab:green"
        )
        """
        ax.set_title(
        f"{title}\nprogressive actions / all actions = {row['tempo']:.2f}"
)
        ax.legend(loc="upper left")

    plt.show()