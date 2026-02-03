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
            if t not in action_types:
                continue

            x, y = _xy(ev.get("location"))
            endx, endy = (np.nan, np.nan)

            if t == "Pass":
                endx, endy = _xy(ev.get("pass", {}).get("end_location"))
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
    """
    Identify progressive actions based on two criteria:
    1) The action must advance the ball by at least p * remaining distance to the opponent's goal.
    2) The action must advance the ball by at least a fixed minimum distance (dx_min).
    
    Parameters:
    - df: DataFrame containing actions, specifically passes with 'x' and 'endx' coordinates.
    - p: Proportion of remaining distance to consider for progressiveness.
    - dx_min: Fixed minimum distance for progressiveness.
    
    Returns:
    - DataFrame with additional columns indicating progressiveness and margin.

    """
    df = df.copy()
    df = df[df["endx"].notna() & df["x"].notna()]

    # forward displacement
    df["dx"] = df["endx"] - df["x"]
    df["remaining"] = 120 - df["x"]

    # only forward actions at all
    df = df[df["dx"] > 0].copy()

    df["thr"] = p * df["remaining"]

    df["is_progressive"] = (
        (df["dx"] >= df["thr"]) &
        (df["dx"] >= dx_min)
    )

    df["margin"] = df["dx"] - df["thr"]

    # diagnostics (useful for plotting / sanity checks)
    df["slack_rel"] = df["dx"] - df["thr"]
    df["slack_abs"] = df["dx"] - dx_min
    df["slack"] = df[["slack_rel", "slack_abs"]].min(axis=1)

    return df