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
PITCH_L, PITCH_W = 120, 80
Y_TOP = 18
Y_MID1 = 30
Y_MID2 = 50
Y_BOT = 62
y_bands = [0, Y_TOP, Y_MID1, Y_MID2, Y_BOT, 80]

MID_DEF_X0, MID_DEF_X1 = 40, 60
MID_ATT_X0, MID_ATT_X1 = 60, 80

MID_Y0, MID_Y1 = 18, 62
# Wide channels are split into N columns (small boxes)
N_WIDE_COLS = 6
x_bins_wide = np.linspace(0, 120, N_WIDE_COLS + 1)  # 0..120 inclusive

# Penalty area (StatsBomb)
PA_L_X0, PA_L_X1 = 0, 18
PA_R_X0, PA_R_X1 = 102, 120
PA_Y0, PA_Y1 = 18, 62

# Your special penalty split using 6-yard box y-bounds
PA_SPLIT_Y1 = 30
PA_SPLIT_Y2 = 50

def rect_poly(x0, x1, y0, y1):
    # polygon as list of (x,y)
    return [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]

def zone_polygon(zone_name):
    # Penalty boxes
    if zone_name == "PenBox_Def_Left":
        return rect_poly(PA_L_X0, PA_L_X1, PA_Y0, PA_SPLIT_Y1)
    if zone_name == "PenBox_Def_Central":
        return rect_poly(PA_L_X0, PA_L_X1, PA_SPLIT_Y1, PA_SPLIT_Y2)
    if zone_name == "PenBox_Def_Right":
        return rect_poly(PA_L_X0, PA_L_X1, PA_SPLIT_Y2, PA_Y1)

    if zone_name == "PenBox_Att_Left":
        return rect_poly(PA_R_X0, PA_R_X1, PA_Y0, PA_SPLIT_Y1)
    if zone_name == "PenBox_Att_Central":
        return rect_poly(PA_R_X0, PA_R_X1, PA_SPLIT_Y1, PA_SPLIT_Y2)
    if zone_name == "PenBox_Att_Right":
        return rect_poly(PA_R_X0, PA_R_X1, PA_SPLIT_Y2, PA_Y1)

        # Center dead zones (only if you have MID_* defined)
    if zone_name == "Center_Dead_Def":
        return rect_poly(MID_DEF_X0, MID_DEF_X1, MID_Y0, MID_Y1)
    if zone_name == "Center_Dead_Att":
        return rect_poly(MID_ATT_X0, MID_ATT_X1, MID_Y0, MID_Y1)

    # Wing zones: top strip and bottom strip split into 6 columns
    if zone_name.startswith("Wing_Left_Zone"):
        k = int(zone_name.replace("Wing_Left_Zone",""))
        x0, x1 = x_bins_wide[k], x_bins_wide[k+1]
        return rect_poly(x0, x1, 0, Y_TOP)

    if zone_name.startswith("Wing_Right_Zone"):
        k = int(zone_name.replace("Wing_Right_Zone",""))
        x0, x1 = x_bins_wide[k], x_bins_wide[k+1]
        return rect_poly(x0, x1, Y_BOT, 80)

    # Pockets (your y bands: 18–30, 30–50, 50–62 ; and x halves)
    pocket_map = {
        "Def_Pocket_Left":   (18, 40, 18, 30),
        "Def_Pocket_Central":(18, 40, 30, 50),
        "Def_Pocket_Right":  (18, 40, 50, 62),
        "Att_Pocket_Left":   (80,102, 18, 30),
        "Att_Pocket_Central":(80,102, 30, 50),
        "Att_Pocket_Right":  (80,102, 50, 62),
    }
    if zone_name in pocket_map:
        x0,x1,y0,y1 = pocket_map[zone_name]
        return rect_poly(x0,x1,y0,y1)

    return None
def get_zone(x: float, y: float) -> str:



    # penalty areas override everything
    in_left_pa  = (PA_L_X0 <= x <= PA_L_X1) and (PA_Y0 <= y <= PA_Y1)
    in_right_pa = (PA_R_X0 <= x <= PA_R_X1) and (PA_Y0 <= y <= PA_Y1)

    if in_left_pa:
        if y < PA_SPLIT_Y1: return "PenBox_Def_Left"
        elif y <= PA_SPLIT_Y2: return "PenBox_Def_Central"
        else: return "PenBox_Def_Right"

    if in_right_pa:
        if y < PA_SPLIT_Y1: return "PenBox_Att_Left"
        elif y <= PA_SPLIT_Y2: return "PenBox_Att_Central"
        else: return "PenBox_Att_Right"

    if (MID_DEF_X0 <= x <= MID_DEF_X1) and (MID_Y0 <= y <= MID_Y1):
        return "Center_Dead_Def"
    if (MID_ATT_X0 <= x <= MID_ATT_X1) and (MID_Y0 <= y <= MID_Y1):
        return "Center_Dead_Att"


    # wide channels
    if y < Y_TOP:
        col = np.digitize([x], x_bins_wide)[0] - 1
        col = int(np.clip(col, 0, N_WIDE_COLS - 1))
        return f"Wing_Left_Zone{col}"

    if y > Y_BOT:
        col = np.digitize([x], x_bins_wide)[0] - 1
        col = int(np.clip(col, 0, N_WIDE_COLS - 1))
        return f"Wing_Right_Zone{col}"

    # central corridor (non-dead)
    if y < 30:
        pocket_y = "Left"
    elif y <= 50:
        pocket_y = "Central"
    else:
        pocket_y = "Right"


    return f"{'Def' if x < 60 else 'Att'}_Pocket_{pocket_y}"

def _load_events_for_match(events_dir: Path, match_id: int):
    p = events_dir / f"{match_id}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())

def _xy(loc):
    # StatsBomb: [x, y] or None
    if not loc or len(loc) < 2:
        return (np.nan, np.nan)
    return (float(loc[0]), float(loc[1]))
def add_t_abs_seconds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["t_abs_sec"] = (
        pd.to_numeric(df["minute"], errors="coerce") * 60
        + pd.to_numeric(df["second"], errors="coerce")
    )

    ts = pd.to_timedelta(df["timestamp"], errors="coerce")
    frac = ts.dt.components["milliseconds"].fillna(0).astype(float) / 1000.0
    df["t_abs_sec"] = df["t_abs_sec"] + frac

    return df

def load_360_data(action_types=("Pass", "Carry", "Dribble"), three_sixty_only=True) -> pd.DataFrame:
    load_dotenv()

    DATA_ROOT = Path(os.environ["EXJOBB_DATA"])
    sb_root = DATA_ROOT / "open-data-master" / "data"
    matches_dir = sb_root / "matches"
    events_dir  = sb_root / "events"
    three_dir   = sb_root / "three-sixty"
    comps = pd.read_json(sb_root / "competitions.json")

    three_match_ids = None
    if three_sixty_only:
        assert three_dir.exists(), f"Missing: {three_dir}"
        three_files = sorted(three_dir.glob("*.json"))
        three_match_ids = set(int(p.stem) for p in three_files)

        

    rows = []
    for mf in sorted(matches_dir.rglob("*.json")):
        comp_id = int(mf.parent.name)
        season_id = int(mf.stem)
        data = json.loads(mf.read_text())
        for m in data:
            if three_sixty_only and (m["match_id"] not in three_match_ids):
                continue
            rows.append({
                    "competition_id": comp_id,
                    "season_id": season_id,
                    "match_id": m["match_id"],
                    "match_date": m.get("match_date"),
                    "home_team": m["home_team"]["home_team_name"],
                    "away_team": m["away_team"]["away_team_name"],
                    "home_team_id": m["home_team"]["home_team_id"],
                    "away_team_id": m["away_team"]["away_team_id"],
       

                })

    matches_360 = pd.DataFrame(rows).merge(
        comps[["competition_id","season_id","competition_name","season_name"]],
        on=["competition_id","season_id"],
        how="left",
    )
    out = []


    match_ids = matches_360["match_id"].unique()

    for mid in tqdm(match_ids, desc="Loading events"):
        evs = _load_events_for_match(events_dir, int(mid))
        for ev in evs:
            t = ev.get("type", {}).get("name")


            x, y = _xy(ev.get("location"))
            endx, endy = (np.nan, np.nan)
            pass_subtype = None
            pass_length = np.nan
            pass_height = None
            pass_body_part = None

            if t == "Pass":
                endx, endy = _xy(ev.get("pass", {}).get("end_location"))
                p = ev.get("pass", {}) or {}
                pass_length = p.get("length", np.nan)
                pass_subtype = (p.get("type", {}) or {}).get("name")  # e.g. "Goal Kick", "Corner", ...
                pass_height = (p.get("height", {}) or {}).get("name")  # Ground Pass / High Pass / etc
                pass_body_part = (p.get("body_part", {}) or {}).get("name")
            elif t == "Carry":
                endx, endy = _xy(ev.get("carry", {}).get("end_location"))
            elif t == "Dribble":
                # Dribble often doesn't have end_location; keep start-only
                pass

            out.append({
                "match_id": int(mid),
                "event_id": ev.get("id"),
                "period": ev.get("period"),
                "minute": ev.get("minute"),
                "second": ev.get("second"),
                "team": ev.get("team", {}).get("name"),
                "team_id": ev.get("team", {}).get("id"),
                "player": ev.get("player", {}).get("name"),
                "player_id": ev.get("player", {}).get("id"),
                "type": t,
                "x": x, "y": y,
                "endx": endx, "endy": endy,
                "play_pattern": ev.get("play_pattern", {}).get("name"),
                "possession": ev.get("possession"),
                "timestamp": ev.get("timestamp"),
                "duration": ev.get("duration", 0.0),
                "possession_team_id": ev.get("possession_team", {}).get("id"),
                "counterpress": bool(ev.get("counterpress", False)),
                "under_pressure": bool(ev.get("under_pressure", False)),
                "pass_length": pass_length,
                "pass_subtype": pass_subtype,
                "pass_height": pass_height,
                "pass_body_part": pass_body_part,


            })

    df = pd.DataFrame(out)
    return df
def plot_team_zone_shares(team_zone_share, team_id, title=None, annotate=True):
    # Build dict zone->share for this team
    tz = team_zone_share[team_zone_share["team_id"] == team_id].copy()
    share_map = dict(zip(tz["zone_end"], tz["share"]))

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.draw(figsize=(12, 8))

    # Choose only zones that exist in this team's map
    zones = sorted(share_map.keys())

    # Normalise for colormap
    max_share = max(share_map.values()) if share_map else 1.0

    for z in zones:
        poly = zone_polygon(z)
        if poly is None:
            continue

        s = share_map[z]
        # shade intensity by share (0..1)
        alpha = 0.15 + 0.75 * (s / max_share) if max_share > 0 else 0.15

        patch = Polygon(poly, closed=True, alpha=alpha, edgecolor="black", linewidth=1)
        ax.add_patch(patch)

        if annotate:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
            ax.text(cx, cy, f"{s*100:.1f}%", ha="center", va="center", fontsize=9)
            


    if title is None:
        title = f"Team {team_id} progressive pass end-zone shares"
    ax.set_title(title)
    plt.show()

def get_progressive_actions(df, p=0.2, dx_min=8.0):
    df = df.copy()
    df = df[df["endx"].notna() & df["x"].notna()]

    df["dx"] = df["endx"] - df["x"]
    df["remaining"] = 120 - df["x"]
    df["thr"] = p * df["remaining"]

    # progressive only if forward + clears both thresholds
    df["is_progressive"] = (
        (df["dx"] > 0) &
        (df["dx"] >= df["thr"]) &
        (df["dx"] >= dx_min)
    )

    return df


def calc_width(df2, match=True):
    """Calculate the average width of passes for each team in each match or overall.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing pass data with columns 'match_id', 'team_id', 'endy'.
    match (bool): If True, calculate average width for each team in each match. If False, calculate overall average width for each team across all matches.

    returns:
    pd.DataFrame: DataFrame with columns 'match_id', 'team_id', and '
    
    
    """
    df_temp = df2.copy()
    df = df_temp[df_temp["is_progressive"] == True]

    # enforce integer IDs early
    for c in ["match_id", "team_id"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64") 
    df["pass_width"] = abs(df["endy"] - 40)
    if match:
        match_width = (
            df.groupby(["match_id", "team_id"])["pass_width"]
            .mean()
            .reset_index(name="mean_width")
        )
        match_width["match_id"] = match_width["match_id"].astype("Int64")
        match_width["team_id"]  = match_width["team_id"].astype("Int64")
        return match_width

    team_width = (
        df.groupby(["team_id", "match_id"])["pass_width"]
        .mean()
        .groupby("team_id")
        .mean()
        .reset_index(name="mean_width")
    )
    team_width["team_id"] = team_width["team_id"].astype("Int64")
    return team_width

def calc_directness(df, match=True):
    """
    Directness = share of actions that are progressive
               = (# progressive) / (# total)

    Requires:
      - match_id
      - team_id
      - is_progressive (bool)
    """
    df_whole = df.copy()

    # enforce integer IDs early
    for c in ["match_id", "team_id"]:
        if c in df_whole.columns:
            df_whole[c] = df_whole[c].astype("Int64")

    group_cols = ["match_id", "team_id"] if match else ["team_id"]

    directness = (
        df_whole.groupby(group_cols)
        .agg(
            n_total=("is_progressive", "size"),
            n_prog=("is_progressive", "sum"),
        )
        .reset_index()
    )

    directness["directness"] = directness["n_prog"] / directness["n_total"]

    return directness


###NEWimport numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def calc_possession_time(df, match=True, max_gap_s=10.0):
    """
    Possession time (seconds) per team from StatsBomb events.

    Counts:
      - the event's own duration
      - plus a capped gap until the next event starts (to approximate continuous possession)
    This avoids the (max-min) span explosion and handles stoppages via gap capping.

    max_gap_s: maximum between-event gap to count as possession (e.g. 8–15 seconds).
    """
    df = df.copy()

    # --- Required columns
    if "timestamp" not in df.columns or "period" not in df.columns:
        raise ValueError("Need 'timestamp' and 'period' columns.")
    if "match_id" not in df.columns:
        raise ValueError("Need 'match_id' column.")

    # --- IDs
    df["match_id"] = df["match_id"].astype("Int64")

    # possession team id
    if "possession_team_id" not in df.columns:
        if "possession_team" in df.columns:
            df["possession_team_id"] = df["possession_team"].apply(
                lambda x: x.get("id") if isinstance(x, dict) else np.nan
            )
        else:
            raise ValueError("Need possession_team_id (or possession_team dict) in df.")
    df["possession_team_id"] = df["possession_team_id"].astype("Int64")

    # duration
    if "duration" not in df.columns:
        df["duration"] = 0.0
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["duration"] = df["duration"].clip(lower=0.0)

    # --- timestamp within period -> seconds
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.datetime64):
        ts = ts.dt.strftime("%H:%M:%S.%f").str.slice(0, 12)

    t_start = pd.to_timedelta(ts.astype(str), errors="coerce").dt.total_seconds()
    t_start = t_start.fillna(0.0)

    t_end = t_start + df["duration"].to_numpy()

    # --- dynamic per-period offsets per match (includes stoppage)
    per_len = (
        pd.DataFrame({"match_id": df["match_id"], "period": df["period"], "t_end": t_end})
        .groupby(["match_id", "period"], as_index=False)["t_end"].max()
        .sort_values(["match_id", "period"])
    )
    per_len["offset_sec"] = per_len.groupby("match_id")["t_end"].shift(1).fillna(0.0)
    per_len["offset_sec"] = per_len.groupby("match_id")["offset_sec"].cumsum()

    df = df.merge(per_len[["match_id", "period", "offset_sec"]],
                  on=["match_id", "period"], how="left")

    df["t_abs_start"] = t_start + df["offset_sec"].fillna(0.0)
    df["t_abs_end"]   = t_end   + df["offset_sec"].fillna(0.0)

    # --- Sort before taking next-event deltas
    df = df.sort_values(["match_id", "t_abs_start"]).reset_index(drop=True)

    # next event start within match
    df["t_next_start"] = df.groupby("match_id")["t_abs_start"].shift(-1)

    # gap AFTER current event ends until next event starts
    gap = (df["t_next_start"] - df["t_abs_end"]).clip(lower=0.0)
    gap = gap.clip(upper=max_gap_s).fillna(0.0)

    # possession increment = event duration + capped gap
    poss_inc =   df["duration"].to_numpy() #gap.to_numpy() +
    df["poss_inc"] = poss_inc

    # aggregate to team
    if match:
        out = (df.groupby(["match_id", "possession_team_id"], as_index=False)["poss_inc"].sum()
                 .rename(columns={"possession_team_id": "team_id", "poss_inc": "possession_seconds"}))
    else:
        out = (df.groupby(["possession_team_id"], as_index=False)["poss_inc"].sum()
                 .rename(columns={"possession_team_id": "team_id", "poss_inc": "possession_seconds"}))

    out["team_id"] = out["team_id"].astype("Int64")
    return out


def calc_possession_share(df):
    """
    Possession share per team per match based on estimated possession_seconds.
    """
    poss_time = calc_possession_time(df, match=True)

    total = (
        poss_time.groupby("match_id")["possession_seconds"]
                 .sum()
                 .reset_index(name="match_possession_seconds")
    )

    out = poss_time.merge(total, on="match_id", how="left")
    out["possession_share"] = out["possession_seconds"] / out["match_possession_seconds"].replace(0, np.nan)
    return out[["match_id", "team_id", "possession_seconds", "possession_share"]]

def calc_tempo(df, match=True):
    """
    Tempo = number of possession-maintaining on-ball actions per minute of possession.

    Uses:
      - calc_possession_time(df, max_gap_s, match) for possession minutes

    Included action types:
      - Pass, Carry, Dribble

    Required columns in df:
      - match_id, team_id, type
      - minute, second, period, possession
    """
    df = df.copy()

    # enforce integer IDs early (same style as your other funcs)
    for c in ["match_id", "team_id"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")

    group_cols = ["match_id", "team_id"] if match else ["team_id"]

    # 1) count tempo actions
    df_actions = df[df["type"].isin(["Pass"])].copy()

    action_counts = (
        df_actions.groupby(group_cols)
        .size()
        .reset_index(name="n_actions")
    )

    # 2) possession time (seconds) using your function
    poss_time = calc_possession_time(df, match=match)

    # 3) tempo = actions / possession minutes
    out = action_counts.merge(poss_time, on=group_cols, how="left")
    out["possession_minutes"] = out["possession_seconds"] / 60.0
    out["tempo"] = out["n_actions"] / out["possession_minutes"].replace(0, np.nan)

    return out


PRESS_EVENTS = {
    "Pressure",
    "Interception",
    "Duel",
    "50/50",
    "Block",
    "Ball Recovery",
    "Foul Committed",
    # optional:
    "Dribbled Past",
}
def calc_pressing_intensity(df_all_events: pd.DataFrame) -> pd.DataFrame:
    """
    Pressing intensity = number of pressing actions while defending
    per minute of opponent possession.

    Returns columns:
      match_id, team_id, n_press_actions, opp_possession_minutes, press_intensity
    """
    df = df_all_events.copy()

    # regular play only (recommended for stability)
    #df = df[df["play_pattern"].fillna("Regular Play") == "Regular Play"].copy()

    # pressing actions
    press = df[df["type"].isin(PRESS_EVENTS)].copy()

    # only when opponent owns the possession
    press = press[press["possession_team_id"].notna()].copy()
    press = press[press["possession_team_id"] != press["team_id"]].copy()

    grp = ["match_id", "team_id"]
    counts = press.groupby(grp).size().reset_index(name="n_press_actions")

    # opponent possession minutes = (match total - own possession)
    poss = calc_possession_time(df, match=True)  # match_id, team_id, possession_seconds
    poss["opp_possession_seconds"] = (
        poss.groupby("match_id")["possession_seconds"].transform("sum") - poss["possession_seconds"]
    )
    poss["opp_possession_minutes"] = poss["opp_possession_seconds"] / 60.0

    out = counts.merge(poss[["match_id", "team_id", "opp_possession_minutes"]], on=grp, how="left")
    out["press_intensity"] = out["n_press_actions"] / out["opp_possession_minutes"].replace(0, np.nan)
    return out
def calc_pressing_height(
    df_all_events: pd.DataFrame,
    high_line_x: float = 60.0,
    very_high_line_x: float = 80.0,
    min_press_actions: int = 10,
) -> pd.DataFrame:
    """
    Pressing height features from press-event x locations.

    Returns columns:
      match_id, team_id,
      n_press_actions,
      press_height_mean_x, press_height_median_x, press_height_std_x,
      n_high, press_height_share_high,
      n_very_high, press_height_share_very_high
    """
    df = df_all_events.copy()

    # regular play only
    df = df[df["play_pattern"].fillna("Regular Play") == "Regular Play"].copy()

    press = df[df["type"].isin(PRESS_EVENTS)].copy()

    # only when defending
    press = press[press["possession_team_id"].notna()].copy()
    press = press[press["possession_team_id"] != press["team_id"]].copy()

    # require x
    press = press[press["x"].notna()].copy()

    grp = ["match_id", "team_id"]
    press["is_high"] = press["x"] >= high_line_x
    press["is_very_high"] = press["x"] >= very_high_line_x

    agg = (
        press.groupby(grp)
             .agg(
                 n_press_actions=("x", "size"),
                 press_height_mean_x=("x", "mean"),
                 press_height_median_x=("x", "median"),
                 press_height_std_x=("x", "std"),
                 n_high=("is_high", "sum"),
                 n_very_high=("is_very_high", "sum"),
             )
             .reset_index()
    )

    agg["press_height_share_high"] = agg["n_high"] / agg["n_press_actions"].replace(0, np.nan)
    agg["press_height_share_very_high"] = agg["n_very_high"] / agg["n_press_actions"].replace(0, np.nan)

    # stability: if too few press actions, ratios + moments are noisy
    mask = agg["n_press_actions"] < min_press_actions
    cols_to_nan = [
        "press_height_mean_x", "press_height_median_x", "press_height_std_x",
        "press_height_share_high", "press_height_share_very_high"
    ]
    agg.loc[mask, cols_to_nan] = np.nan

    return agg
def calc_pass_length(
    df: pd.DataFrame,
    match: bool = True,
    exclude_restarts: bool = True,
) -> pd.DataFrame:
    """
    Pass length features (mean + std) using StatsBomb 'pass_length' extracted at load time.

    Required columns:
      - match_id
      - team_id
      - type == "Pass"
      - pass_length
    Optional:
      - pass_subtype (for excluding restarts)

    Returns:
      match=True: match_id, team_id, n_passes, mean_pass_length, std_pass_length
      match=False: team_id, n_passes, mean_pass_length, std_pass_length
    """
    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = df["match_id"].astype("Int64")
    if "team_id" in df.columns:
        df["team_id"] = df["team_id"].astype("Int64")

    passes = df[(df["type"] == "Pass") & (df["pass_length"].notna())].copy()

    if exclude_restarts and "pass_subtype" in passes.columns:
        exclude = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
        passes = passes[~passes["pass_subtype"].isin(exclude)].copy()

    grp = ["match_id", "team_id"] if match else ["team_id"]

    out = (
        passes.groupby(grp)
              .agg(
                  n_passes=("pass_length", "size"),
                  mean_pass_length=("pass_length", "mean"),
                  std_pass_length=("pass_length", "std"),
              )
              .reset_index()
    )

    out["team_id"] = out["team_id"].astype("Int64")
    return out
