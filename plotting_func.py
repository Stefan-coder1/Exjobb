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
import functions

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




def plot_two_matches(df, a, b, mode="tempo"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)

    for ax, row, title in zip(
        axes,
        [a, b],
        [f"Most {mode}", f"Least {mode}"]
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
        f"{title}\nprogressive actions / all actions = {row[mode]:.2f}"
)
        ax.legend(loc="upper left")

    plt.show()


def plot_two_pressing_intensity_curves(df, row_a, row_b, max_opp_poss_min=15, labels=None):
    if labels is None:
        labels = ["A", "B"]
    def _get_press_curve(df, row, max_opp_poss_min=None):
        sub = df[
            (df["match_id"] == row["match_id"]) &
            (df["possession_team_id"] != row["team_id"])
        ].copy()

        sub["t"] = pd.to_timedelta(sub["timestamp"])
        sub = sub.sort_values("t")

        # opponent possession time (only within the same possession id)
        sub["t_prev"] = sub["t"].shift(1)
        same_poss = sub["possession"] == sub["possession"].shift(1)
        dt = (sub["t"] - sub["t_prev"]).dt.total_seconds().clip(lower=0)
        sub["opp_poss_dt"] = np.where(same_poss, dt, 0.0)
        sub["cum_opp_poss"] = sub["opp_poss_dt"].cumsum()

        # pressing actions (count only events made by the team of interest)
        sub["is_press"] = (
            (sub["team_id"] == row["team_id"]) &
            (sub["type"].isin(functions.PRESS_EVENTS))
        ).astype(int)
        sub["cum_press"] = sub["is_press"].cumsum()

        if max_opp_poss_min is not None:
            sub = sub[sub["cum_opp_poss"] <= max_opp_poss_min * 60].copy()

        x = sub["cum_opp_poss"].to_numpy() / 60.0
        y = sub["cum_press"].to_numpy()
        return x, y



    x1, y1 = _get_press_curve(df, row_a, max_opp_poss_min=max_opp_poss_min)
    x2, y2 = _get_press_curve(df, row_b, max_opp_poss_min=max_opp_poss_min)

    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, lw=2, label=labels[0])
    plt.plot(x2, y2, lw=2, label=labels[1])

    plt.xlabel("Opponent possession time (minutes)")
    plt.ylabel("Cumulative pressing actions")
    plt.title(f"Pressing intensity comparison (first {max_opp_poss_min} opp poss min)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_two_pressing_height_distributions(df, row_a, row_b, labels=None, bins=30):
    if labels is None:
        labels = ["Team A", "Team B"]

    def get_press_x(row):
        press = df[
            (df["match_id"] == row["match_id"]) &
            (df["team_id"] == row["team_id"]) &
            (df["possession_team_id"] != row["team_id"]) &
            (df["type"].isin(functions.PRESS_EVENTS)) &
            (df["x"].notna())
        ]
        return press["x"].values

    x_a = get_press_x(row_a)
    x_b = get_press_x(row_b)
    mean_a = x_a.mean()
    mean_b = x_b.mean()

    plt.figure(figsize=(8, 6))
    plt.hist(x_a, bins=bins, density=True, alpha=0.6, label=labels[0])
    plt.hist(x_b, bins=bins, density=True, alpha=0.6, label=labels[1])

    plt.axvline(60, color="k", linestyle="--", label="Opponent half")


    plt.axvline(mean_a, color="blue", linestyle="-", linewidth=2)
    plt.axvline(mean_b, color="orange", linestyle="-", linewidth=2)


    plt.xlabel("x-position")
    plt.ylabel("Density")
    plt.title("Pressing height comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()
