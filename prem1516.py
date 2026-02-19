import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from mplsoccer import Pitch, VerticalPitch





def load_competitions(sb_data_root: Path) -> pd.DataFrame:
    comp_path = sb_data_root / "competitions.json"
    comps = json.loads(comp_path.read_text(encoding="utf-8"))
    return pd.DataFrame(comps)

TARGET = [
    ("England", "Premier League"),
  #  ("Spain", "La Liga"),
  #  ("Italy", "Serie A"),
  #  ("Germany", "1. Bundesliga"),
]

def pick_competitions_1516(comps):
    selected = []

    for country, comp in TARGET:
        sel = comps[
            (comps["country_name"] == country) &
            (comps["competition_name"] == comp) &
            (comps["season_name"] == "2015/2016")
        ]

        if sel.empty:
            raise ValueError(f"Missing: {country} {comp} 2015/2016")

        selected.append(sel.iloc[0])

    return pd.DataFrame(selected)


def load_matches(sb_data_root: Path, competition_id: int, season_id: int) -> pd.DataFrame:
    p = sb_data_root / "matches" / str(competition_id) / f"{season_id}.json"
    matches = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame(matches)

def _safe_id(x):
    if isinstance(x, dict) and "id" in x: return x["id"]
    return np.nan

def _safe_name(x):
    if isinstance(x, dict) and "name" in x: return x["name"]
    return None

def flatten_events_for_match(sb_data_root: Path, match_row: dict) -> pd.DataFrame:
    match_id = match_row["match_id"]
    p = sb_data_root / "events" / f"{match_id}.json"
    ev = json.loads(p.read_text(encoding="utf-8"))

    rows = []
    for e in ev:
        loc = e.get("location", None)
        x = loc[0] if isinstance(loc, list) and len(loc) >= 2 else np.nan
        y = loc[1] if isinstance(loc, list) and len(loc) >= 2 else np.nan

        # end locations (pass/carry)
        endx = endy = np.nan
        pass_length = np.nan
        pass_subtype = None

        if "pass" in e and e["pass"] is not None:
            end = e["pass"].get("end_location", None)
            if isinstance(end, list) and len(end) >= 2:
                endx, endy = end[0], end[1]
            pass_length = e["pass"].get("length", np.nan)
            pass_subtype = _safe_name(e["pass"].get("type"))
        elif "carry" in e and e["carry"] is not None:
            end = e["carry"].get("end_location", None)
            if isinstance(end, list) and len(end) >= 2:
                endx, endy = end[0], end[1]

        rows.append({
            "match_id": match_id,
            "competition_id": match_row["competition"]["competition_id"] if isinstance(match_row.get("competition"), dict) else match_row.get("competition_id"),
            "season_id": match_row["season"]["season_id"] if isinstance(match_row.get("season"), dict) else match_row.get("season_id"),
            "competition_name": match_row.get("competition", {}).get("competition_name", None) if isinstance(match_row.get("competition"), dict) else None,
            "season_name": match_row.get("season", {}).get("season_name", None) if isinstance(match_row.get("season"), dict) else None,

            "type": _safe_name(e.get("type")),
            "play_pattern": _safe_name(e.get("play_pattern")),
            "team_id": _safe_id(e.get("team")),
            "team_name": _safe_name(e.get("team")),
            "possession": e.get("possession", np.nan),
            "possession_team_id": _safe_id(e.get("possession_team")),
            "possession_team_name": _safe_name(e.get("possession_team")),

            "minute": e.get("minute", np.nan),
            "second": e.get("second", np.nan),
            "timestamp": e.get("timestamp", None),  
            "duration": e.get("duration", np.nan),  
            "period" : e.get("period", np.nan),

            "x": x, "y": y,
            "endx": endx, "endy": endy,

            "pass_length": pass_length,
            "pass_subtype": pass_subtype,
        })

    return pd.DataFrame(rows)

def load_all_events_1516(sb_data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comps = load_competitions(sb_data_root)
    picked = pick_competitions_1516(comps)

    all_matches = []
    for _, r in picked.iterrows():
        m = load_matches(sb_data_root, int(r["competition_id"]), int(r["season_id"]))
        # enrich for convenience
        m["competition_name"] = r["competition_name"]
        m["season_name"] = r["season_name"]
        all_matches.append(m)

    matches_df = pd.concat(all_matches, ignore_index=True)

    # Load events
    event_dfs = []
    for _, mr in matches_df.iterrows():
        event_dfs.append(flatten_events_for_match(sb_data_root, mr.to_dict()))

    events_df = pd.concat(event_dfs, ignore_index=True)

    # create a "league" label that matches your normalization bucket
    events_df["league_season"] = events_df["competition_name"].fillna("") + " | " + events_df["season_name"].fillna("")
    matches_df["league_season"] = matches_df["competition_name"].fillna("") + " | " + matches_df["season_name"].fillna("")

    return comps, matches_df, events_df
from functions import (
    calc_width, calc_directness, calc_tempo,
    calc_pressing_height, calc_pressing_intensity,
    calc_pass_length, get_progressive_actions, calc_possession_time
)
from tqdm import tqdm

def build_team_match_features_1516(sb_data_root: Path) -> pd.DataFrame:
    comps = load_competitions(sb_data_root)
    picked = pick_competitions_1516(comps)

    all_feature_rows = []

    for _, r in picked.iterrows():

        matches = load_matches(
            sb_data_root,
            int(r["competition_id"]),
            int(r["season_id"])
        )

        league_label = r["competition_name"] + " | " + r["season_name"]

        for _, match in tqdm(
            matches.iterrows(),
            total=len(matches),
            desc=f"{league_label}",
            leave=False
        ):
            events = flatten_events_for_match(sb_data_root, match.to_dict())

            actions = get_progressive_actions(events)

            f_width = calc_width(actions, match=True)
            f_dir   = calc_directness(actions, match=True)
            f_tempo = calc_tempo(events, match=True)
            f_ph    = calc_pressing_height(events)
            f_pi    = calc_pressing_intensity(events)
            f_pl    = calc_pass_length(events, match=True)

            merged = (
                f_width
                .merge(f_dir, on=["match_id","team_id"], how="outer")
                .merge(f_tempo, on=["match_id","team_id"], how="outer")
                .merge(f_ph, on=["match_id","team_id"], how="outer")
                .merge(f_pi, on=["match_id","team_id"], how="outer")
                .merge(f_pl, on=["match_id","team_id"], how="outer")
            )
            team_map = events[["team_id", "team_name"]].drop_duplicates()
            merged = merged.merge(team_map, on="team_id", how="left")

            merged["league_season"] = league_label
            all_feature_rows.append(merged)

            # free memory
            del events, actions, f_width, f_dir, f_tempo, f_ph, f_pi, f_pl, merged

    return pd.concat(all_feature_rows, ignore_index=True)

