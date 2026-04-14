import re

import torch
import torch.nn.functional as F

@torch.no_grad()
def sample_categorical(logits, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def mask_logits(logits_vec, forbid_ids):
    logits_vec = logits_vec.clone()
    for i in forbid_ids:
        if i is not None and 0 <= i < logits_vec.numel():
            logits_vec[i] = -1e9
    return logits_vec

def mask_logits_by_allowed(logits_vec, allowed_mask):
    """
    logits_vec: [V]
    allowed_mask: [V] bool
    """
    logits_vec = logits_vec.clone()
    logits_vec[~allowed_mask] = -1e9
    return logits_vec
@torch.no_grad()
def generate_one_fixed(
    model, c_vec, zone_code, min_code,
    type2id, role2id, sz2id, ez2id, out2id, dt2id, term2id,
    start_xy,
    T=40, temperature=0.9, device="cpu", z_override=None, valid_outcome_mask=None
):
    model.eval()

    c  = torch.tensor(c_vec, dtype=torch.float32, device=device).unsqueeze(0)
    zc = torch.tensor([zone_code], dtype=torch.long, device=device)
    mc = torch.tensor([min_code], dtype=torch.long, device=device)
    cond = model.make_cond(c, zc, mc)

    z = z_override if z_override is not None else torch.randn((1, model.zdim), device=device)


    END_ID   = type2id["END"]
    PASS_ID  = type2id["Pass"]
    CARRY_ID = type2id["Carry"]

    PAD_ROLE = role2id["PAD"]
    PAD_SZ   = sz2id["PAD"]
    PAD_EZ   = ez2id["PAD"]
    PAD_OUT  = out2id["PAD"]
    PAD_DT   = dt2id["PAD"]
    NA_TERM  = term2id.get("NA_TERM", 0)

    NA_END_EZ = ez2id.get("NA_END", None)

    # Storage for generated (committed) tokens
    x_gen = torch.empty((1, T, 7), dtype=torch.long, device=device)
    x_gen[:, :, 0] = PAD_ROLE
    x_gen[:, :, 1] = 0         # doesn't matter; will be overwritten
    x_gen[:, :, 2] = PAD_SZ
    x_gen[:, :, 3] = PAD_EZ
    x_gen[:, :, 4] = PAD_OUT
    x_gen[:, :, 5] = PAD_DT
    x_gen[:, :, 6] = NA_TERM

    dxy_gen = torch.zeros((1, T, 2), dtype=torch.float32, device=device)
    xy_gen = torch.zeros((1, T, 2), dtype=torch.float32, device=device)

    # BOS vector (you used 0 in training)
    BOS = torch.zeros((7,), dtype=torch.long, device=device)

    prev_ez = None

    for t in range(T):
        # Build shifted-prefix inputs (THIS is the generation equivalent of teacher forcing)
        x_in  = torch.empty_like(x_gen)
        dxy_in = torch.zeros_like(dxy_gen)
        xy_in = torch.zeros((1, T, 2), dtype=torch.float32, device=device)
        xy_in[:, 0, :] = torch.tensor(start_xy, dtype=torch.float32, device=device)

           

        # default all future to PAD-ish
        x_in[:, :, 0] = PAD_ROLE
        x_in[:, :, 1] = 0
        x_in[:, :, 2] = PAD_SZ
        x_in[:, :, 3] = PAD_EZ
        x_in[:, :, 4] = PAD_OUT
        x_in[:, :, 5] = PAD_DT
        x_in[:, :, 6] = NA_TERM

        x_in[:, 0, :] = BOS
        if t > 0:
            x_in[:, 1:t+1, :] = x_gen[:, :t, :]
            dxy_in[:, 1:t+1, :] = dxy_gen[:, :t, :]
            xy_in[:, 1:t+1, :] = xy_gen[:, :t, :]

        # Decode with correct argument order
        logits = model.decode(x_in=x_in, z=z, cond=cond, dxy_in=dxy_in, xy_in=xy_in, ez_for_dxy=None)

        # --- sample type first
        ty = sample_categorical(logits["type"][0, t], temperature)

        # role
        if ty == END_ID:
            role = PAD_ROLE
        else:
            role_logits = mask_logits(logits["role"][0, t], [PAD_ROLE])
            role = sample_categorical(role_logits, temperature)

        # start zone as state
        if t == 0:
            sz = sample_categorical(logits["sz"][0, t], temperature)
            xy_prev = torch.tensor(start_xy, dtype=torch.float32, device=device)
        else:
            xy_prev = xy_gen[0, t-1]
            if prev_ez is None or prev_ez == PAD_EZ or (NA_END_EZ is not None and prev_ez == NA_END_EZ):
                sz = x_gen[0, t-1, 2].item()
            else:
                sz = prev_ez

        # end zone only for pass/carry
        if ty in (PASS_ID, CARRY_ID):
            ez_logits = logits["ez"][0, t]
            forbid = [PAD_EZ]
            if NA_END_EZ is not None:
                forbid.append(NA_END_EZ)
            ez = sample_categorical(mask_logits(ez_logits, forbid), temperature)

        else:
            ez = NA_END_EZ if NA_END_EZ is not None else PAD_EZ

        # out / dt (you can later mask these by type; leaving as-is for now)
        # out / dt
        if valid_outcome_mask is not None:
            allowed_out = valid_outcome_mask[ty].to(logits["out"].device)   # [n_out]
            out_logits = mask_logits_by_allowed(logits["out"][0, t], allowed_out)
            out = sample_categorical(out_logits, temperature)
        else:
            out = sample_categorical(logits["out"][0, t], temperature)

        dt = sample_categorical(logits["dt"][0, t], temperature)

        # term only for END
        if ty == END_ID:
            sz = PAD_SZ; ez = PAD_EZ; out = PAD_OUT; dt = PAD_DT
            term = sample_categorical(logits["term"][0, t], temperature)
        else:
            term = NA_TERM

        # Commit token
        x_gen[0, t] = torch.tensor([role, ty, sz, ez, out, dt, term], device=device)

        # Commit dxy
        if ty in (PASS_ID, CARRY_ID):
            
            dxy = logits["dxy"][0, t]
            dxy_gen[0, t] = dxy
            xy_gen[0, t] = xy_prev + dxy
        else:
            dxy_gen[0, t] = 0.0
            xy_gen[0, t] = xy_prev

        prev_ez = ez
        if ty == END_ID:
            break

    return {
        "start_xy": tuple(start_xy),
        "role": x_gen[0, :, 0].tolist(),
        "type": x_gen[0, :, 1].tolist(),
        "sz":   x_gen[0, :, 2].tolist(),
        "ez":   x_gen[0, :, 3].tolist(),
        "out":  x_gen[0, :, 4].tolist(),
        "dt":   x_gen[0, :, 5].tolist(),
        "term": x_gen[0, :, 6].tolist(),
        "dxy":  dxy_gen[0].cpu().tolist(),
        "xy_end": xy_gen[0].cpu().tolist(),
    }
def pretty_print_seq_with_xy_dxy(decoded, max_rows=60):
    """
    decoded = output of decode_seq_dxy(gen)
    which contains:
        {
            "start_xy": (x, y),
            "steps": [...]
        }
    """

    start_xy = decoded.get("start_xy", None)
    steps = decoded["steps"]

    prev_xy = start_xy

    for t, st in enumerate(steps[:max_rows]):
        ty = st["type"]

        if ty == "END":
            print(f"{t:02d} END  term={st['term']}")
            break

        dx, dy = st.get("dxy", (0.0, 0.0))
        xy_end = st.get("xy_end", None)

        # determine start/end for display
        x0, y0 = prev_xy if prev_xy is not None else (None, None)

        if xy_end is not None:
            x1, y1 = xy_end
        else:
            x1 = x0 + dx if x0 is not None else None
            y1 = y0 + dy if y0 is not None else None

        # format coordinates nicely
        if x0 is not None:
            xy_str = f"({x0:5.1f},{y0:5.1f}) -> ({x1:5.1f},{y1:5.1f})"
        else:
            xy_str = "(None)"

        print(
            f"{t:02d} {st['role']:3s}  {st['type']:14s}  "
            f"{st['sz']:18s} -> {st['ez']:18s}  "
            f"out={st['out']:10s}  dt={st['dt']:5s}  term={st['term']:10s}  "
            f"{xy_str}  dxy=({dx:+.2f},{dy:+.2f})"
        )

        prev_xy = (x1, y1)
import numpy as np
from mplsoccer import Pitch

def decode_seq_dxy(gen, iddict=None):
    """
    Decode generated ids to readable step dicts, while preserving
    start_xy and xy_end if present.
    """
    if iddict is None:
        raise ValueError("iddict is required for decode_seq_dxy to map ids to strings")
    id2type = iddict["id2type"]
    id2role = iddict["id2role"]
    id2sz = iddict["id2sz"]
    id2ez = iddict["id2ez"]
    id2out = iddict["id2out"]
    id2dt = iddict["id2dt"]
    id2term = iddict["id2term"]
    T = len(gen["type"])
    steps = []

    start_xy = tuple(map(float, gen["start_xy"])) if "start_xy" in gen else None
    xy_end_all = gen.get("xy_end", None)

    for t in range(T):
        ty = id2type.get(gen["type"][t], "UNK")
        if ty == "PAD":
            break

        dx, dy = (0.0, 0.0)
        if "dxy" in gen:
            dx, dy = gen["dxy"][t]

        xy_end = None
        if xy_end_all is not None and t < len(xy_end_all):
            xy_end = tuple(map(float, xy_end_all[t]))

        steps.append({
            "role": id2role[gen["role"][t]],
            "type": ty,
            "sz":   id2sz[gen["sz"][t]],
            "ez":   id2ez[gen["ez"][t]],
            "out":  id2out[gen["out"][t]],
            "dt":   id2dt[gen["dt"][t]],
            "term": id2term[gen["term"][t]],
            "dxy":  (float(dx), float(dy)),
            "xy_end": xy_end,
        })

        if ty == "END":
            break

    return {
        "start_xy": start_xy,
        "steps": steps,
    }


def clamp_to_pitch(x, y):
    x = float(np.clip(x, 0.0, 120.0))
    y = float(np.clip(y, 0.0, 80.0))
    return x, y


def clamp_to_zone_bbox(x, y, zone_name):
    bb = cvaefirst.zone_bbox(zone_name)
    if bb is None:
        return x, y
    x0, x1, y0, y1 = bb
    x = float(np.clip(x, x0, x1))
    y = float(np.clip(y, y0, y1))
    return x, y


def zone_point(zone_name, rng=None, mode="centroid"):
    if rng is None:
        rng = np.random.default_rng(0)
    bb = cvaefirst.zone_bbox(zone_name)
    if bb is None:
        return None
    x0, x1, y0, y1 = bb
    if mode == "centroid":
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    if mode == "random":
        return (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
    raise ValueError("mode must be 'centroid' or 'random'")


def safe_zone_point(zone_name, rng, mode="centroid", fallback="pitch_center"):
    p = zone_point(zone_name, rng=rng, mode=mode)
    if p is not None:
        return p

    if fallback == "pitch_random":
        return (float(rng.uniform(0, 120)), float(rng.uniform(0, 80)))

    return (60.0, 40.0)


def plot_sequence_from_dxy(
    decoded,
    seed=0,
    title=None,
    mode_start="centroid",
    enforce_start_zone=False,
    enforce_end_zone=False,
    clip_pitch=True,
    fallback_start="pitch_center",
    use_xy_end=True,   # NEW: prefer model-returned xy_end if available
):
    """
    decoded = output from decode_seq_dxy(gen), i.e.
    {
        "start_xy": (x, y),
        "steps": [...]
    }

    Behavior:
    - Uses decoded["start_xy"] as the first absolute point if available
    - Then rolls forward continuously from previous end
    - For Pass/Carry:
        * if use_xy_end and step["xy_end"] exists -> use that
        * else compute end from start + dxy
    """
    rng = np.random.default_rng(seed)
    pitch = Pitch(pitch_type="statsbomb")
    fig, ax = pitch.draw(figsize=(10, 6))
    if title:
        ax.set_title(title)

    steps = decoded["steps"]
    start_xy = decoded.get("start_xy", None)

    prev_end = None

    for i, st in enumerate(steps):
        ty = st.get("type")
        if ty == "END":
            break

        role = st.get("role", "ATT")
        is_def = (role == "DEF")
        alpha = 0.5 if is_def else 0.9
        lw = 2.0 if not is_def else 1.5

        sz = st.get("sz")
        ez = st.get("ez")

        # --- START POINT ---
        if i == 0 and start_xy is not None:
            x0, y0 = map(float, start_xy)
            if enforce_start_zone:
                x0, y0 = clamp_to_zone_bbox(x0, y0, sz)
        elif prev_end is not None:
            x0, y0 = prev_end
            if enforce_start_zone:
                x0, y0 = clamp_to_zone_bbox(x0, y0, sz)
        else:
            # fallback only if no start_xy exists
            x0, y0 = safe_zone_point(sz, rng=rng, mode=mode_start, fallback=fallback_start)

        if clip_pitch:
            x0, y0 = clamp_to_pitch(x0, y0)

        # --- END POINT ---
        if ty in ("Pass", "Carry"):
            if use_xy_end and st.get("xy_end") is not None:
                x1, y1 = st["xy_end"]
            else:
                dx, dy = st.get("dxy", (0.0, 0.0))
                x1, y1 = x0 + float(dx), y0 + float(dy)

            if enforce_end_zone:
                x1, y1 = clamp_to_zone_bbox(x1, y1, ez)

            if clip_pitch:
                x1, y1 = clamp_to_pitch(x1, y1)

        else:
            x1, y1 = x0, y0

        # --- DRAW ---
        if ty == "Pass":
            pitch.arrows(
                x0, y0, x1, y1,
                ax=ax,
                linewidth=lw / 3,
                headwidth=3,
                headlength=3,
                alpha=alpha,
            )
        elif ty == "Carry":
            pitch.lines(
                x0, y0, x1, y1,
                ax=ax,
                linestyle="dotted",
                linewidth=lw,
                alpha=alpha,
            )
        elif ty == "Shot":
            shot_out = st.get("out", "NA")

            if shot_out == "Goal":
                # sample inside goal mouth
                sx = 120.0
                sy = float(rng.uniform(36.0, 44.0))

            elif shot_out in ("Saved", "Saved to Post"):
                sx = 120.0
                sy = float(rng.uniform(36.0, 44.0))

            elif shot_out == "Post":
                sx = 120.0
                sy = 36.0 if y0 < 40 else 44.0

            elif shot_out == "Blocked":
                # blocked before goal
                sx = min(x0 + rng.uniform(4, 10), 118)
                sy = float(np.clip(y0 + rng.normal(0, 2), 0, 80))

            elif shot_out in ("Off T", "Wayward"):
                # miss left/right of goal
                sx = 120.0
                sy = 30.0 if y0 < 40 else 50.0

            else:
                sx, sy = 120.0, 40.0

            pitch.arrows(
                x0, y0, sx, sy,
                ax=ax,
                linewidth=lw,
                headwidth=3,
                headlength=3,
                alpha=alpha,
                color="red",
            )
        elif ty in ("Interception", "Block"):
            pitch.scatter(
                [x0], [y0],
                ax=ax,
                s=120,
                marker="x",
                linewidths=2,
                zorder=5,
                alpha=alpha,
            )
        else:
            pitch.scatter([x0], [y0], ax=ax, s=30, alpha=alpha)

        pitch.annotate(str(i), (x0, y0), ax=ax, fontsize=8)
        prev_end = (x1, y1)

    return fig, ax

import numpy as np
import pandas as pd
import torch

def generate_many_compare(
    model,
    c_vec_a, c_vec_b,
    zone_code, min_code,
    type2id, role2id, sz2id, ez2id, out2id, dt2id, term2id,
    n_samples=100,
    T=40,
    temperature=0.9,
    device="cpu",
    start_xy=(60.0, 40.0),   # NEW
):
    # same latent draws for both conditions = fairer comparison
    z_bank = [torch.randn((1, model.zdim), device=device) for _ in range(n_samples)]

    seqs_a = []
    seqs_b = []

    for z in z_bank:
        seq_a = generate_one_fixed(
            model=model,
            c_vec=c_vec_a,
            zone_code=zone_code,
            min_code=min_code,
            type2id=type2id,
            role2id=role2id,
            sz2id=sz2id,
            ez2id=ez2id,
            out2id=out2id,
            dt2id=dt2id,
            term2id=term2id,
            T=T,
            temperature=temperature,
            device=device,
            z_override=z,
            start_xy=start_xy,   # NEW
        )

        seq_b = generate_one_fixed(
            model=model,
            c_vec=c_vec_b,
            zone_code=zone_code,
            min_code=min_code,
            type2id=type2id,
            role2id=role2id,
            sz2id=sz2id,
            ez2id=ez2id,
            out2id=out2id,
            dt2id=dt2id,
            term2id=term2id,
            T=T,
            temperature=temperature,
            device=device,
            z_override=z,
            start_xy=start_xy,   # NEW
        )

        seqs_a.append(seq_a)
        seqs_b.append(seq_b)

    return seqs_a, seqs_b


def valid_steps(seq, type2id):
    end_id = type2id["END"]
    steps = []
    for t, ty in enumerate(seq["type"]):
        if ty == end_id:
            break
        steps.append(t)
    return steps


def sequence_metrics(seq, type2id, id2type=None, default_start_xy=(60.0, 40.0)):
    PASS_ID = type2id["Pass"]
    CARRY_ID = type2id["Carry"]

    steps = valid_steps(seq, type2id)

    n_steps = len(steps)
    n_pass = 0
    n_carry = 0
    n_other = 0

    abs_dy_actions = []
    y_positions = []

    # NEW: use actual generated start_xy if present
    start_xy = tuple(seq.get("start_xy", default_start_xy))
    x, y = start_xy

    xy_end_all = seq.get("xy_end", None)

    for t in steps:
        ty = seq["type"][t]

        # Prefer actual rolled/generated endpoint if available
        if xy_end_all is not None and t < len(xy_end_all):
            x1, y1 = seq["xy_end"][t]
            dx = x1 - x
            dy = y1 - y
        else:
            dx, dy = seq["dxy"][t]
            x1, y1 = x + dx, y + dy

        if ty == PASS_ID:
            n_pass += 1
            abs_dy_actions.append(abs(dy))
            x, y = x1, y1
            y_positions.append(y)

        elif ty == CARRY_ID:
            n_carry += 1
            abs_dy_actions.append(abs(dy))
            x, y = x1, y1
            y_positions.append(y)

        else:
            n_other += 1

    mean_abs_dy = float(np.mean(abs_dy_actions)) if abs_dy_actions else np.nan
    mean_width_from_center = float(np.mean([abs(v - 40.0) for v in y_positions])) if y_positions else np.nan

    return {
        "start_x": float(start_xy[0]),
        "start_y": float(start_xy[1]),
        "length": n_steps,
        "n_pass": n_pass,
        "n_carry": n_carry,
        "n_other": n_other,
        "pass_share": n_pass / n_steps if n_steps > 0 else np.nan,
        "carry_share": n_carry / n_steps if n_steps > 0 else np.nan,
        "mean_abs_dy": mean_abs_dy,
        "mean_width_from_center": mean_width_from_center,
        "final_x": float(x),
        "final_y": float(y),
    }


def summarize_group(seqs, type2id, default_start_xy=(60.0, 40.0)):
    rows = [sequence_metrics(seq, type2id, default_start_xy=default_start_xy) for seq in seqs]
    df = pd.DataFrame(rows)
    return df, df.mean(numeric_only=True)


def decode_dataset_seq(dataset, idx, iddict):
    id2type = iddict["id2type"]
    id2role = iddict["id2role"]
    id2sz = iddict["id2sz"]
    id2ez = iddict["id2ez"]
    id2out = iddict["id2out"]
    id2dt = iddict["id2dt"]
    id2term = iddict["id2term"]
    x, dxy_t, xy0_t, xy1_t, cond, zone_code, min_code, mask, length = dataset[idx]
    L = int(length.item())

    steps = []
    start_xy = tuple(map(float, xy0_t[0].tolist())) if L > 0 else None

    for t in range(L):
        role_id, type_id, sz_id, ez_id, out_id, dt_id, term_id = x[t].tolist()

        ty = id2type.get(type_id, "UNK")
        if ty == "PAD":
            break

        steps.append({
            "role": id2role.get(role_id, "UNK"),
            "type": ty,
            "sz": id2sz.get(sz_id, "UNK"),
            "ez": id2ez.get(ez_id, "UNK"),
            "out": id2out.get(out_id, "UNK"),
            "dt": id2dt.get(dt_id, "UNK"),
            "term": id2term.get(term_id, "UNK"),
            "dxy": tuple(map(float, dxy_t[t].tolist())),
            "xy_end": tuple(map(float, xy1_t[t].tolist())),
        })

        if ty == "END":
            break

    return {
        "start_xy": start_xy,
        "steps": steps,
        "cond": cond,
        "zone_code": int(zone_code.item()),
        "min_code": int(min_code.item()),
    }