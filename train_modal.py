from pathlib import Path
import json
import math
from typing import Any

import modal
from tqdm import tqdm


app = modal.App("exjobb-cvae")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "mplsoccer",
        "tqdm",
        "pyzmq",          # needed because cvaefirst imports zmq
        "python-dotenv",  # needed because functions.py imports dotenv
    )
    .add_local_python_source(
        "cvaefirst",
        "functions",
    )
)

# Create once if needed:
# modal volume create exjobb-data
volume = modal.Volume.from_name("exjobb-data")


def _to_numpy(x: Any):
    """Convert tensors/lists to numpy-like arrays without assuming storage type."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
    except Exception:
        pass
    return x


def _build_ez_bb(ez2id: dict, device: str = "cpu"):
    """
    Build end-zone bounding boxes from the current functions.zone_polygon implementation.

    Returns a tensor [n_ez, 4] with columns [x0, x1, y0, y1]. Special/non-spatial
    classes such as PAD or NA_END receive a full-pitch box so the zone penalty does
    not crash when those ids are present.
    """
    import numpy as np
    import torch
    import functions

    n_ez = max(int(v) for v in ez2id.values()) + 1
    boxes = np.zeros((n_ez, 4), dtype="float32")

    id2ez = {int(v): k for k, v in ez2id.items()}
    full_pitch = [0.0, 120.0, 0.0, 80.0]

    for idx in range(n_ez):
        zone_name = id2ez.get(idx, "")
        poly = functions.zone_polygon(zone_name)
        if poly is None:
            boxes[idx] = full_pitch
            continue

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes[idx] = [min(xs), max(xs), min(ys), max(ys)]

    return torch.tensor(boxes, dtype=torch.float32, device=device)


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 180,
    gpu="A10G",
)
def train(
    epochs: int = 100,
    patience: int = 12,
    early_stop_after: int = 20,
    batch_size: int = 512,
    dataset_blob_name: str = "dataset_blob_big4.pt",
    dataset_meta_name: str = "dataset_meta_big4.pt",
):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    import cvaefirst

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    artifact_dir = Path("/data")
    ckpt_dir = artifact_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print("Loading dataset artifacts...")
    blob_path = artifact_dir / dataset_blob_name
    meta_path = artifact_dir / dataset_meta_name
    print("Blob:", blob_path)
    print("Meta:", meta_path)

    blob = torch.load(blob_path, map_location="cpu")
    meta = torch.load(meta_path, map_location="cpu")

    # Old context setup: Cz + start zone + minute bin only.
    seq_ids = blob["seq_ids"]
    Cz = _to_numpy(blob["Cz"])
    start_zone_code = _to_numpy(blob["start_zone_code"])
    minute_code = _to_numpy(blob["minute_code"])

    train_idx = _to_numpy(blob["train_idx"])
    val_idx = _to_numpy(blob["val_idx"])
    test_idx = _to_numpy(blob["test_idx"])

    type2id = meta["type2id"]
    sz2id = meta["sz2id"]
    ez2id = meta["ez2id"]
    out2id = meta["out2id"]
    dt2id = meta["dt2id"]
    term2id = meta["term2id"]
    role2id = meta["role2id"]
    T = int(meta["T"])

    valid_outcome_mask = meta["valid_outcome_mask"]
    if not isinstance(valid_outcome_mask, torch.Tensor):
        valid_outcome_mask = torch.tensor(valid_outcome_mask, dtype=torch.bool)

    # Safety checks for the old context setup.
    assert len(seq_ids) == len(Cz), (len(seq_ids), len(Cz))
    assert len(seq_ids) == len(start_zone_code), (len(seq_ids), len(start_zone_code))
    assert len(seq_ids) == len(minute_code), (len(seq_ids), len(minute_code))

    cdim = int(Cz.shape[1])
    n_zones = int(np.max(start_zone_code)) + 1
    n_mins = int(np.max(minute_code)) + 1

    print(f"N sequences: {len(seq_ids)}")
    print(f"Cz shape: {Cz.shape}")
    print(f"n_zones={n_zones}, n_mins={n_mins}, T={T}")
    print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    EZ_BB = _build_ez_bb(ez2id, device=device)

    ds_all = cvaefirst.PossessionDataset(
        seq_ids,
        Cz,
        start_zone_code,
        minute_code,
        T=T,
    )

    train_ds = Subset(ds_all, train_idx)
    val_ds = Subset(ds_all, val_idx)
    test_ds = Subset(ds_all, test_idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train batches: {len(train_dl)}")
    print(f"Val batches: {len(val_dl)}")
    print(f"Test batches: {len(test_dl)}")

    model = cvaefirst.SeqCVAE(
        n_role=len(role2id),
        n_types=len(type2id),
        n_sz=len(sz2id),
        n_ez=len(ez2id),
        n_out=len(out2id),
        n_dt=len(dt2id),
        n_term=len(term2id),
        n_zones=n_zones,
        n_minbins=n_mins,
        emb=32,
        hidden=256,
        zdim=32,
        cdim=cdim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    LAMBDA_DXY = 1.0
    LAMBDA_ZONE = 3.0
    LAMBDA_SOFT = 3.0

    best_score = math.inf
    best_score_epoch = -1
    best_median = math.inf
    best_median_epoch = -1
    epochs_without_improvement = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_total": [],
        "val_main": [],
        "val_term": [],
        "val_dxy_loss": [],
        "val_kld": [],
        "val_zone": [],
        "val_soft": [],
        "val_align": [],
        "val_median_m": [],
        "val_mean_m": [],
        "beta": [],
        "ss_prob": [],
    }

    def save_ckpt(path: Path, epoch: int, extra: dict):
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "type2id": type2id,
            "sz2id": sz2id,
            "ez2id": ez2id,
            "out2id": out2id,
            "dt2id": dt2id,
            "term2id": term2id,
            "role2id": role2id,
            "config": {
                "emb": 32,
                "hidden": 256,
                "zdim": 32,
                "cdim": cdim,
                "T": T,
                "n_zones": n_zones,
                "n_mins": n_mins,
                "batch_size": batch_size,
                "lambda_dxy": LAMBDA_DXY,
                "lambda_zone": LAMBDA_ZONE,
                "lambda_soft": LAMBDA_SOFT,
                "dataset_blob_name": dataset_blob_name,
                "dataset_meta_name": dataset_meta_name,
            },
            **extra,
        }
        torch.save(ckpt, path)

    epoch_bar = tqdm(range(epochs), desc="Training", position=0)

    for epoch in epoch_bar:
        beta = min(1.0, epoch / 15.0)
        ss_prob = max(0.0, min(0.3, (epoch - 5.0) / 50.0))

        print(f"\nStarting epoch {epoch + 1}/{epochs} | beta={beta:.3f} | ss_prob={ss_prob:.3f}")

        train_loss = cvaefirst.train_one_epoch(
            model,
            train_dl,
            opt,
            device,
            type2id,
            beta=beta,
            lambda_dxy=LAMBDA_DXY,
            lambda_soft=LAMBDA_SOFT,
            ez_bb=EZ_BB,
            ez2id=ez2id,
            lambda_zone=LAMBDA_ZONE,
            out2id=out2id,
            ss_prob=ss_prob,
            valid_outcome_mask=valid_outcome_mask,
        )

        val_stats = cvaefirst.eval_one_epoch(
            model,
            val_dl,
            device,
            type2id,
            beta=beta,
            lambda_dxy=LAMBDA_DXY,
            lambda_soft=LAMBDA_SOFT,
            ez_bb=EZ_BB,
            ez2id=ez2id,
            lambda_zone=LAMBDA_ZONE,
            out2id=out2id,
            valid_outcome_mask=valid_outcome_mask,
        )

        meters = cvaefirst.validate_dxy_meters(
            model=model,
            loader=val_dl,
            device=device,
            type2id=type2id,
            max_batches=None,
        )

        median_m = float(meters.get("median_m", float("nan")))
        mean_m = float(meters.get("mean_m", float("nan")))

        val_main = float(val_stats.get("main", float("nan")))
        val_total = float(val_stats.get("total", val_main))
        score = val_total

        epoch_bar.set_postfix(
            {
                "train": f"{float(train_loss):.2f}",
                "val": f"{score:.2f}",
                "med_m": f"{median_m:.2f}",
            }
        )

        print(
            f"ep {epoch:03d} beta={beta:.2f} "
            f"train={float(train_loss):.3f} "
            f"val_total={val_total:.3f} "
            f"val_main={val_main:.3f} "
            f"val_kld={float(val_stats.get('kld', float('nan'))):.3f} "
            f"val_dxy={float(val_stats.get('dxy', float('nan'))):.3f} "
            f"val_zone={float(val_stats.get('zone', 0.0)):.3f} "
            f"val_soft={float(val_stats.get('soft', 0.0)):.3f} "
            f"val_align={float(val_stats.get('align', 0.0)):.3f} "
            f"dxy_med={median_m:.2f}m "
            f"dxy_mean={mean_m:.2f}m"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_total"].append(val_total)
        history["val_main"].append(val_main)
        history["val_term"].append(float(val_stats.get("term", float("nan"))))
        history["val_dxy_loss"].append(float(val_stats.get("dxy", float("nan"))))
        history["val_kld"].append(float(val_stats.get("kld", float("nan"))))
        history["val_zone"].append(float(val_stats.get("zone", float("nan"))))
        history["val_soft"].append(float(val_stats.get("soft", float("nan"))))
        history["val_align"].append(float(val_stats.get("align", float("nan"))))
        history["val_median_m"].append(median_m)
        history["val_mean_m"].append(mean_m)
        history["beta"].append(beta)
        history["ss_prob"].append(ss_prob)

        # Save history every epoch so partial runs still leave usable results.
        with open(artifact_dir / "training_history_big4_oldcontext.json", "w") as f:
            json.dump(history, f, indent=2)

        # Main checkpoint: best validation total/main loss.
        if not math.isnan(score) and score < best_score - 1e-3:
            best_score = score
            best_score_epoch = epoch
            epochs_without_improvement = 0
            save_ckpt(
                ckpt_dir / "best_cvae_big4_oldcontext_by_val.pt",
                epoch,
                {"best_score": best_score, "best_score_name": "val_total_or_main"},
            )
            print(f"Saved best validation checkpoint at epoch {epoch} with score {best_score:.4f}")
        else:
            if epoch >= early_stop_after:
                epochs_without_improvement += 1

        # Secondary checkpoint: best median displacement error.
        if not math.isnan(median_m) and median_m < best_median - 0.05:
            best_median = median_m
            best_median_epoch = epoch
            save_ckpt(
                ckpt_dir / "best_cvae_big4_oldcontext_by_dxy.pt",
                epoch,
                {"best_median_m": best_median, "best_score_name": "val_median_dxy_m"},
            )
            print(f"Saved best dxy checkpoint at epoch {epoch} with median {best_median:.2f}m")

        if epoch >= early_stop_after and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"No validation improvement for {patience} epochs after epoch {early_stop_after}."
            )
            break

    final_ckpt = ckpt_dir / "cvae_v1_big4_oldcontext_final.pt"
    save_ckpt(
        final_ckpt,
        epoch,
        {
            "best_score": best_score,
            "best_score_epoch": best_score_epoch,
            "best_median_m": best_median,
            "best_median_epoch": best_median_epoch,
        },
    )

    # Persist Modal volume writes.
    try:
        volume.commit()
    except Exception as e:
        print("Volume commit warning:", repr(e))

    print("Saved final model:", final_ckpt)
    print("Best validation epoch:", best_score_epoch, "best score:", best_score)
    print("Best dxy epoch:", best_median_epoch, "best median meters:", best_median)

    return {
        "best_score_epoch": best_score_epoch,
        "best_score": best_score,
        "best_median_epoch": best_median_epoch,
        "best_median_m": best_median,
        "history_path": str(artifact_dir / "training_history_big4_oldcontext.json"),
        "best_val_checkpoint": str(ckpt_dir / "best_cvae_big4_oldcontext_by_val.pt"),
        "best_dxy_checkpoint": str(ckpt_dir / "best_cvae_big4_oldcontext_by_dxy.pt"),
        "final_checkpoint": str(final_ckpt),
    }


@app.local_entrypoint()
def main(
    epochs: int = 100,
    patience: int = 12,
    early_stop_after: int = 20,
    batch_size: int = 512,
):
    result = train.remote(
        epochs=epochs,
        patience=patience,
        early_stop_after=early_stop_after,
        batch_size=batch_size,
    )
    print(result)
