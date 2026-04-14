
import numpy as np
from zmq import device
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#Even larger loader
import json
from pathlib import Path
import pandas as pd
import os
import functions
import re



class PossessionDataset(Dataset):
    def __init__(self, seq_ids, cond_vecs, zone_code, minute_code, T: int):
        self.seqs = seq_ids
        self.cond = torch.as_tensor(cond_vecs, dtype=torch.float32)  # Cz [N,12]
        self.zone = torch.as_tensor(zone_code, dtype=torch.long)     # [N]
        self.minb = torch.as_tensor(minute_code, dtype=torch.long)   # [N]
        self.T = int(T)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]

        def pad(x, pad_id=0):
            x = x[: self.T]
            if len(x) < self.T:
                x = x + [pad_id] * (self.T - len(x))
            return torch.tensor(x, dtype=torch.long)
        def pad_xy(xy):
            xy = xy[: self.T]
            if len(xy) < self.T:
                xy = xy + [[0.0, 0.0]] * (self.T - len(xy))
            return torch.tensor(xy, dtype=torch.float32)  # [T,2]
        

        def pad_dxy(dxy):
            dxy = dxy[: self.T]
            if len(dxy) < self.T:
                dxy = dxy + [[0.0, 0.0]] * (self.T - len(dxy))
            return torch.tensor(dxy, dtype=torch.float32)  # [T,2]
        
        role_ids = pad(s["role"], 0)
        type_ids = pad(s["type"], 0)
        sz_ids   = pad(s["sz"],   0)
        ez_ids   = pad(s["ez"],   0)
        out_ids  = pad(s["out"],  0)
        dt_ids   = pad(s["dt"],   0)
        term_ids = pad(s["term"], 0)

        dxy_t = pad_dxy(s["dxy"])  
        xy0_t = pad_xy(s["xy0"])
        xy1_t = pad_xy(s["xy1"])

        mask = (type_ids != 0).float()
        length = torch.tensor(int(mask.sum().item()), dtype=torch.long)
        #xy1_t = torch.nan_to_num(xy1_t, nan=0.0, posinf=0.0, neginf=0.0)
        #dxy_t = torch.nan_to_num(dxy_t, nan=0.0, posinf=0.0, neginf=0.0)

        
        x = torch.stack([role_ids, type_ids, sz_ids, ez_ids, out_ids, dt_ids, term_ids], dim=-1)

        # NEW: also return zone/minute context codes
        return x,  dxy_t, xy0_t, xy1_t, self.cond[i], self.zone[i], self.minb[i], mask, length


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,T,V], target: [B,T], mask: [B,T] in {0,1}
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), target.reshape(B * T), reduction="none")
    loss = loss.reshape(B, T) * mask
    return loss.sum() / (mask.sum() + 1e-8)
def masked_ce_with_type_constraints(logits, target, type_ids, mask, valid_outcome_mask):
    B, T, V = logits.shape
    allowed = valid_outcome_mask[type_ids]   # [B,T,V]

    gold_allowed = allowed.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B,T]
    bad = mask.bool() & (~gold_allowed)
    if bad.any():
        bad_idx = bad.nonzero(as_tuple=False)[:10]
        print("Disallowed gold targets:")
        for b, t in bad_idx:
            print(
                "type_id =", int(type_ids[b, t]),
                "target_id =", int(target[b, t]),
            )
        raise RuntimeError("Gold target masked out by valid_outcome_mask")

    neg_inf = torch.finfo(logits.dtype).min
    logits_masked = logits.masked_fill(~allowed, neg_inf)

    loss = F.cross_entropy(
        logits_masked.reshape(B * T, V),
        target.reshape(B * T),
        reduction="none"
    ).reshape(B, T)

    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)

def kld(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())


class SeqCVAE(nn.Module):
    def __init__(
        self,
        n_types: int,
        n_sz: int,
        n_ez: int,
        n_out: int,
        n_dt: int,
        n_term: int,
        n_zones: int,        # NEW
        n_minbins: int,  
        n_role: int,
        emb: int = 32,
        hidden: int = 256,
        zdim: int = 32,
        cdim: int = 6,
        zone_emb_dim: int = 8,                 
        min_emb_dim: int = 4,                  
    ):

        super().__init__()
        self.cdim = cdim
        self.zdim = zdim
    

        # embeddings
        self.type_emb = nn.Embedding(n_types, emb, padding_idx=0)
        self.sz_emb   = nn.Embedding(n_sz,   emb, padding_idx=0)
        self.ez_emb   = nn.Embedding(n_ez,   emb, padding_idx=0)
        self.out_emb  = nn.Embedding(n_out,  emb, padding_idx=0)
        self.dt_emb   = nn.Embedding(n_dt,   emb, padding_idx=0)
        self.term_emb = nn.Embedding(n_term, emb, padding_idx=0)
        self.zone_ctx_emb = nn.Embedding(n_zones, zone_emb_dim)
        self.min_ctx_emb  = nn.Embedding(n_minbins, min_emb_dim)
        self.role_emb = nn.Embedding(n_role, emb, padding_idx=0)

        self.c_eff_dim = cdim + zone_emb_dim + min_emb_dim  
        self.dxy_emb = nn.Linear(2, emb)   # or 16, or whatever you like
        self.xy_emb = nn.Linear(2, emb)
# and update decoder input dimension:

        


        self.ez_dxy_emb = nn.Embedding(n_ez, 16, padding_idx=0)  # 16 is enough

        in_dim = emb * 7 

        # encoder (packed)
        self.enc_rnn = nn.GRU(in_dim + self.c_eff_dim, hidden, batch_first=True)
        self.to_mu   = nn.Linear(hidden, zdim)
        self.to_logv = nn.Linear(hidden, zdim)

        # decoder (teacher forcing, un-packed is fine)
        
        self.dec_rnn = nn.GRU(in_dim + self.c_eff_dim + zdim + emb + emb, hidden, batch_first=True)

        # heads

        self.h_dxy = nn.Sequential(
            nn.Linear(hidden + 16, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )
        self.dxy_scale = 1.0  # set later; or keep =1 if your dxy is already normalized
        self.h_type = nn.Linear(hidden, n_types)
        self.h_sz   = nn.Linear(hidden, n_sz)
        self.h_ez   = nn.Linear(hidden, n_ez)
        self.h_out  = nn.Linear(hidden, n_out)
        self.h_dt   = nn.Linear(hidden, n_dt)
        self.h_term = nn.Linear(hidden, n_term)
        self.h_role = nn.Linear(hidden, n_role)

    def embed_step(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,7]
        assert x.shape[-1] == 7, x.shape
        role, t, sz, ez, out, dt, term = x.unbind(dim=-1)
        tok_emb = torch.cat(
            [
                self.role_emb(role),
                self.type_emb(t),
                self.sz_emb(sz),
                self.ez_emb(ez),
                self.out_emb(out),
                self.dt_emb(dt),
                self.term_emb(term),
            ],
            dim=-1,
        )
        

        return torch.cat([tok_emb], dim=-1)
    def make_cond(self, c: torch.Tensor, zone_code: torch.Tensor, min_code: torch.Tensor):
        # c: [B, cdim], zone_code/min_code: [B]
        zemb = self.zone_ctx_emb(zone_code)
        memb = self.min_ctx_emb(min_code)
        return torch.cat([c, zemb, memb], dim=-1)  # [B, c_eff_dim] 
    def encode(self, x, c, lengths, dxy_true=None):
        B, T, _ = x.shape
        cond_rep = c.unsqueeze(1).expand(B, T, c.shape[-1])

        x_emb = self.embed_step(x)           # [B,T,in_dim]
        parts = [x_emb, cond_rep]

        

        inp = torch.cat(parts, dim=-1)       # must match enc_rnn input dim!

        packed = pack_padded_sequence(inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.enc_rnn(packed)
        h = h.squeeze(0)
        return self.to_mu(h), self.to_logv(h)

    def reparam(self, mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, x_in, z, cond, dxy_in, xy_in, ez_for_dxy=None):
        B, T, _ = x_in.shape

        e = self.embed_step(x_in)                    # [B,T,in_dim]
        d = self.dxy_emb(dxy_in)                     # [B,T,emb]
        xy = self.xy_emb(xy_in)                      # [B,T,emb]

        cond_rep = cond.unsqueeze(1).expand(B, T, cond.shape[-1])
        z_rep    = z.unsqueeze(1).expand(B, T, z.shape[-1])

        inp = torch.cat([e, d, xy, cond_rep, z_rep], dim=-1)

        out, _ = self.dec_rnn(inp)

        ez_logits = self.h_ez(out)

        if ez_for_dxy is None:
            ez_for_dxy = ez_logits.argmax(dim=-1)

        ez_ctx = self.ez_dxy_emb(ez_for_dxy)           # [B,T,16]
        dxy    = self.h_dxy(torch.cat([out, ez_ctx], dim=-1)) * self.dxy_scale

        return {
            "role": self.h_role(out),
            "type": self.h_type(out),
            "sz":   self.h_sz(out),
            "ez":   ez_logits,
            "out":  self.h_out(out),
            "dt":   self.h_dt(out),
            "term": self.h_term(out),
            "dxy":  dxy,
        }
    

    def forward(self, x, c, zone_code, min_code, xy0_t, xy1_t, lengths, ss_prob=0.0, ss_temp=1.0):
        cond = self.make_cond(c, zone_code, min_code)
        cond = torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)

        dxy_true  = xy1_t - xy0_t
        dxy_clean = torch.nan_to_num(dxy_true, nan=0.0, posinf=0.0, neginf=0.0)
        xy0_clean = torch.nan_to_num(xy0_t, nan=0.0, posinf=0.0, neginf=0.0)
        xy1_clean = torch.nan_to_num(xy1_t, nan=0.0, posinf=0.0, neginf=0.0)

        xy_in = xy1_clean.clone()
        xy_in[:, 1:, :] = xy1_clean[:, :-1, :]   # previous true end xy
        xy_in[:, 0, :]  = xy0_clean[:, 0, :]     # possession start xy

        mu, logv = self.encode(x, cond, lengths, dxy_true=None)
        z = self.reparam(mu, logv)

        B, T, _ = x.shape
        device = x.device
        lengths_d = lengths.to(device)

        # teacher-forced shifted inputs
        x_in = x.clone()
        x_in[:, 1:, :] = x[:, :-1, :]
        x_in[:, 0, :]  = 0

        dxy_in = dxy_clean.clone()
        dxy_in[:, 1:, :] = dxy_clean[:, :-1, :]
        dxy_in[:, 0, :]  = 0.0

        _, _, _, ez_t, _, _, _ = x.unbind(dim=-1)
        ez_for_dxy = torch.zeros_like(ez_t)
        ez_for_dxy[:, 1:] = ez_t[:, :-1]
        ez_for_dxy[:, 0]  = 0

        # ---- Pass 1: pure teacher forcing
        logits_tf = self.decode(x_in=x_in, z=z, cond=cond, dxy_in=dxy_in, xy_in=xy_in, ez_for_dxy=ez_for_dxy)

        if ss_prob <= 0.0:
            return logits_tf, mu, logv

        # ---- Build mixed inputs for Pass 2
        # We will overwrite the "previous step" positions 1..T-1
        # For each timestep i (1..T-1), x_in[:, i] represents tokens from step i-1.
        # So we use predictions from step i-1.

        # mask of positions to overwrite: [B, T]
        t_idx = torch.arange(T, device=device).view(1, T).expand(B, T)
        valid_pos = (t_idx < lengths_d.view(B, 1)) & (t_idx > 0)   # only 1..len-1
        use_model = (torch.rand(B, T, device=device) < ss_prob) & valid_pos
        

        # get predicted tokens from previous step (shifted by +1 in time)
        def argmax_shift(logits_key):
            pred = logits_tf[logits_key].argmax(dim=-1)  # [B,T]
            pred_shift = torch.zeros_like(pred)
            pred_shift[:, 1:] = pred[:, :-1]
            return pred_shift

        pred_role = argmax_shift("role")
        pred_type = argmax_shift("type")
        pred_sz   = argmax_shift("sz")
        pred_ez   = argmax_shift("ez")
        pred_out  = argmax_shift("out")
        pred_dt   = argmax_shift("dt")
        pred_term = argmax_shift("term")

        # overwrite selected positions in x_in
        x_in_m = x_in.clone()
        x_in_m[..., 0] = torch.where(use_model, pred_role, x_in_m[..., 0])
        x_in_m[..., 1] = torch.where(use_model, pred_type, x_in_m[..., 1])
        x_in_m[..., 2] = torch.where(use_model, pred_sz,   x_in_m[..., 2])
        x_in_m[..., 3] = torch.where(use_model, pred_ez,   x_in_m[..., 3])
        x_in_m[..., 4] = torch.where(use_model, pred_out,  x_in_m[..., 4])
        x_in_m[..., 5] = torch.where(use_model, pred_dt,   x_in_m[..., 5])
        x_in_m[..., 6] = torch.where(use_model, pred_term, x_in_m[..., 6])

        # overwrite dxy_in similarly (shifted)
        dxy_pred = logits_tf["dxy"].detach()               # [B,T,2]
        dxy_shift = torch.zeros_like(dxy_pred)
        dxy_shift[:, 1:, :] = dxy_pred[:, :-1, :]

        dxy_in_m = dxy_in.clone()
        dxy_in_m = torch.where(use_model.unsqueeze(-1), dxy_shift, dxy_in_m)

        # overwrite ez_for_dxy (shifted predicted ez)
        ez_for_dxy_m = ez_for_dxy.clone()
        ez_for_dxy_m = torch.where(use_model, pred_ez, ez_for_dxy_m)
        # predicted end positions from pass 1
        xy1_pred = (xy0_clean + dxy_pred).detach()   # [B,T,2]

        # shifted version so step t gets previous predicted end xy from step t-1
        xy_shift = xy_in.clone()
        xy_shift[:, 1:, :] = xy1_pred[:, :-1, :]
        xy_shift[:, 0, :]  = xy0_clean[:, 0, :]   # keep real possession start
        xy_in_m = xy_in.clone()
        xy_in_m = torch.where(use_model.unsqueeze(-1), xy_shift, xy_in_m)

        # ---- Pass 2: decode on mixed history (this is the one you backprop through)
        logits = self.decode(x_in=x_in_m, z=z, cond=cond, dxy_in=dxy_in_m,xy_in=xy_in_m, ez_for_dxy=ez_for_dxy_m)

        return logits, mu, logv

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B,T,2]
    mask: [B,T] in {0,1}
    """
    m = mask.bool()
    if m.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_v = pred[m]      # [N,2]
    targ_v = target[m]    # [N,2]
    # If you still suspect data issues, you can harden this:
    # pred_v = torch.nan_to_num(pred_v, nan=0.0, posinf=0.0, neginf=0.0)
    # targ_v = torch.nan_to_num(targ_v, nan=0.0, posinf=0.0, neginf=0.0)

    return F.smooth_l1_loss(pred_v, targ_v, reduction="mean")

def masked_beta_nll(u_true, ax, bx, ay, by, mask, eps=1e-4):
    # u_true: [B,T,2] in [0,1]
    ux = u_true[...,0].clamp(eps, 1-eps)
    uy = u_true[...,1].clamp(eps, 1-eps)

    # log Beta PDF per dim:
    # log p(u) = (a-1)log u + (b-1)log(1-u) - (lgamma(a)+lgamma(b)-lgamma(a+b))
    logBx = torch.lgamma(ax) + torch.lgamma(bx) - torch.lgamma(ax + bx)
    logBy = torch.lgamma(ay) + torch.lgamma(by) - torch.lgamma(ay + by)

    logpx = (ax-1)*torch.log(ux) + (bx-1)*torch.log(1-ux) - logBx
    logpy = (ay-1)*torch.log(uy) + (by-1)*torch.log(1-uy) - logBy

    nll = -(logpx + logpy)  # [B,T]
    nll = nll * mask
    return nll.sum() / (mask.sum() + 1e-8)
import torch.nn.functional as F

def bbox_outside_penalty(xy, bb):
    """
    xy: [N,2]
    bb: [N,4] with x0,x1,y0,y1
    Returns [N] penalty: 0 if inside, >0 if outside (L1 distance outside).
    """
    x, y = xy[:, 0], xy[:, 1]
    x0, x1, y0, y1 = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    px = F.relu(x0 - x) + F.relu(x - x1)
    py = F.relu(y0 - y) + F.relu(y - y1)
    return px + py




def soft_zone_dist_from_bboxes(xy_bt2: torch.Tensor, ez_bb_k4: torch.Tensor, alpha: float = 2.0, eps: float = 1e-8):
    """
    xy_bt2: [B,T,2]
    ez_bb_k4: [K,4] with x0,x1,y0,y1 (aligned with ez class ids)
    returns q: [B,T,K] where q sums to 1 over K
    """
    B, T, _ = xy_bt2.shape
    K = ez_bb_k4.shape[0]
    device = xy_bt2.device

    xy = xy_bt2.reshape(B * T, 2)                          # [BT,2]
    bb = ez_bb_k4.to(device).unsqueeze(0).expand(B * T, K, 4)  # [BT,K,4]

    x = xy[:, 0].unsqueeze(-1)  # [BT,1]
    y = xy[:, 1].unsqueeze(-1)  # [BT,1]

    x0 = bb[..., 0]  # [BT,K]
    x1 = bb[..., 1]
    y0 = bb[..., 2]
    y1 = bb[..., 3]

    # outside penalty per zone (same as your bbox_outside_penalty but batched over K)
    px = F.relu(x0 - x) + F.relu(x - x1)
    py = F.relu(y0 - y) + F.relu(y - y1)
    pen = px + py  # [BT,K]

    # turn penalty into a soft membership score
    scores = torch.exp(-alpha * pen)  # [BT,K]
    q = (scores + eps) / (scores.sum(dim=-1, keepdim=True) + eps * K)
    return q.view(B, T, K)

def compute_loss(
    logits: dict,
    x: torch.Tensor,
    dxy_t: torch.Tensor,                 # kept for now (unused)
    xy0_t: torch.Tensor,
    xy1_t: torch.Tensor,
    mask: torch.Tensor,
    type_end_id: int,
    type_pass_id: int,
    type_carry_id: int,
    mu: torch.Tensor,
    logv: torch.Tensor,
    beta: float = 1.0,
    lambda_dxy: float = 1.0,
    lambda_soft: float = 0.1,
    lambda_zone: float = 0.2,
    ez_bb: torch.Tensor | None = None,
    ez_pad_id: int | None = None,
    ez_na_id: int | None = None,
    valid_outcome_mask=None,
):
    role_t, type_t, sz_t, ez_t, out_t, dt_t, term_t = x.unbind(dim=-1)

    loss_main = (
        masked_ce(logits["role"], role_t, mask) +
        masked_ce(logits["type"], type_t, mask) +
        masked_ce(logits["sz"],   sz_t,   mask) +
        masked_ce(logits["ez"],   ez_t,   mask) +
        masked_ce_with_type_constraints(logits["out"], out_t, type_t, mask, valid_outcome_mask) +
        #masked_ce(logits["out"],  out_t,  mask) + #ANTINGEN DENNA ELLER ASNDRA
        masked_ce(logits["dt"],   dt_t,   mask)
    )
    end_mask = (type_t == type_end_id).float() * mask
    loss_term = masked_ce(logits["term"], term_t, end_mask) if end_mask.sum() > 0 else torch.tensor(0.0, device=x.device)

    # True dxy

    # True dxy


    # base mask: pass/carry and non-pad
    # True displacement
    
    loss_align = torch.tensor(0.0, device=x.device)

    # Only Pass/Carry
    is_move = ((type_t == type_pass_id) | (type_t == type_carry_id))

    # Coordinates must be finite
    coord_ok = torch.isfinite(xy0_t).all(dim=-1) & torch.isfinite(xy1_t).all(dim=-1)

    # Final mask
    dxy_mask = is_move.float() * mask * coord_ok.float()


    # OPTIONAL: do NOT use has_end, or make it tiny
    # has_end = (dxy_true.abs().sum(dim=-1) > 1e-9)
    # dxy_mask = dxy_mask * has_end.float()

    dxy_true = xy1_t - xy0_t
    dxy_true = torch.nan_to_num(dxy_true, nan=0.0, posinf=0.0, neginf=0.0)
    loss_dxy = masked_l1(logits["dxy"], dxy_true, dxy_mask)

    # Soft expected zone penalty (only after final dxy_mask)
    loss_soft = torch.tensor(0.0, device=x.device)
    if ez_bb is not None and dxy_mask.sum() > 0:
        p_ez = torch.softmax(logits["ez"], dim=-1)      # [B,T,n_ez]
        xy1_hat = xy0_t + logits["dxy"]                 # [B,T,2]

        bb = ez_bb.view(1, 1, -1, 4).to(x.device).to(dtype=xy0_t.dtype)

        xh, yh = xy1_hat[..., 0].unsqueeze(2), xy1_hat[..., 1].unsqueeze(2)  # [B,T,1]
        x0, x1, y0, y1 = bb[...,0], bb[...,1], bb[...,2], bb[...,3]          # [1,1,n_ez]

        px = F.relu(x0 - xh) + F.relu(xh - x1)
        py = F.relu(y0 - yh) + F.relu(yh - y1)
        pen_all = (px + py).squeeze(2)              # [B,T,n_ez]

        exp_pen = (p_ez * pen_all).sum(dim=-1)      # [B,T]
        loss_soft = (exp_pen * dxy_mask).sum() / (dxy_mask.sum() + 1e-6)
        p_ez = F.softmax(logits["ez"], dim=-1)            # [B,T,K]
        xy1_hat = xy0_t + logits["dxy"]                   # [B,T,2]
        q_xy = soft_zone_dist_from_bboxes(xy1_hat, ez_bb, alpha=2.0)

        eps = 1e-8
        kl_qp = (q_xy.clamp_min(eps) * (q_xy.clamp_min(eps).log() - p_ez.clamp_min(eps).log())).sum(dim=-1)  # [B,T]

        loss_align = (kl_qp * dxy_mask).sum() / dxy_mask.sum().clamp_min(1.0)

        
    




    # final mask

    loss_zone = torch.tensor(0.0, device=x.device)
    if ez_bb is not None and dxy_mask.sum() > 0:
        # safety: device + dtype
        if ez_bb.device != x.device:
            ez_bb = ez_bb.to(x.device)
        ez_bb = ez_bb.to(dtype=xy0_t.dtype)

        zone_mask = dxy_mask.bool()
        if ez_pad_id is not None:
            zone_mask &= (ez_t != ez_pad_id)
        if ez_na_id is not None:
            zone_mask &= (ez_t != ez_na_id)

        if zone_mask.any():
            dxy_pred = logits["dxy"]            # [B,T,2]
            xy_pred_end = xy0_t + dxy_pred      # [B,T,2]

            bb = ez_bb[ez_t]                    # [B,T,4]
            pen = bbox_outside_penalty(xy_pred_end[zone_mask], bb[zone_mask])
            loss_zone = pen.mean()

    loss_kld = kld(mu, logv)
    loss_kld = torch.nan_to_num(loss_kld, nan=0.0, posinf=1e6, neginf=1e6)



    total = loss_main + loss_term + beta*loss_kld + lambda_dxy*loss_dxy + lambda_zone*loss_zone + lambda_soft*loss_soft + loss_align*lambda_soft
    return total, {
        "main": loss_main.detach(),
        "term": loss_term.detach(),
        "dxy":  loss_dxy.detach(),
        "zone": loss_zone.detach(),
        "kld": loss_kld.detach(),
        "soft": lambda_soft*loss_soft.detach(),
        "align": lambda_soft*loss_align.detach(),

    }
    #--- Example training loop skeleton (minimal) ---
from tqdm import tqdm
import torch
import torch.nn as nn

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    type2id,
    beta,
    lambda_dxy=1.0,
    lambda_soft=0.1,
    lambda_zone=0.2,
    ez_bb=None,
    ez2id=None,
    out2id=None,
    valid_outcome_mask=None,
    ss_prob=0.0,
):
    model.train()

    END_ID   = type2id["END"]
    PASS_ID  = type2id["Pass"]
    CARRY_ID = type2id["Carry"]

    ez_pad_id = ez2id.get("PAD") if isinstance(ez2id, dict) else None
    ez_na_id  = ez2id.get("NA_END") if isinstance(ez2id, dict) else None

    if ez_bb is not None:
        ez_bb = ez_bb.to(device)

    running = 0.0
    pbar = tqdm(loader, desc=f"train beta={beta:.2f}", leave=False)

    for batch in pbar:
        # Expect: x, dxy_t, xy0_t, xy1_t, c, zone_code, min_code, mask, lengths
        if len(batch) != 9:
            raise ValueError(
                f"Expected batch of len 9, got {len(batch)}. "
                f"Shapes: {[getattr(b,'shape',None) for b in batch]}"
            )

        x, dxy_t, xy0_t, xy1_t, c, zone_code, min_code, mask, lengths = batch

        x         = x.to(device)
        dxy_t     = dxy_t.to(device)     # can be unused; keep for now
        xy0_t     = xy0_t.to(device)
        xy1_t     = xy1_t.to(device)
        c         = c.to(device)
        zone_code = zone_code.to(device)
        min_code  = min_code.to(device)
        mask      = mask.to(device)
        lengths   = lengths.to(device)
        valid_outcome_mask = valid_outcome_mask.to(device)


        # Your model.forward must match this signature.
        if not torch.isfinite(xy0_t).all():
            # ok, expected sometimes (NaNs)
            pass
        if not torch.isfinite(xy1_t).all():
            pass

        # This MUST be finite (because you nan_to_num it in forward)
        dxy_true = xy1_t - xy0_t
        dxy_clean = torch.nan_to_num(dxy_true, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(dxy_clean).all():
            raise RuntimeError("dxy_clean is still non-finite")
        logits, mu, logv = model(x, c, zone_code, min_code, xy0_t, xy1_t, lengths, ss_prob=ss_prob)

        
        

        # ---- loss ----
        loss, parts = compute_loss(
            logits=logits,
            x=x,
            dxy_t=dxy_t,
            xy0_t=xy0_t,
            xy1_t=xy1_t,
            mask=mask,
            type_end_id=END_ID,
            type_pass_id=PASS_ID,
            type_carry_id=CARRY_ID,
            mu=mu,
            logv=logv,
            beta=beta,
            lambda_dxy=lambda_dxy,
            lambda_soft=lambda_soft,
            lambda_zone=lambda_zone,
            ez_bb=ez_bb,
            ez_pad_id=ez_pad_id,
            ez_na_id=ez_na_id,
            valid_outcome_mask=valid_outcome_mask
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running += float(loss.item())
        pbar.set_postfix(
            loss=float(loss.item()),
            main=float(parts.get("main", 0.0)),
            dxy=float(parts.get("dxy", 0.0)),
            zone=float(parts.get("zone", 0.0)),
            soft=float(parts.get("soft", 0.0)),
            kld=float(parts.get("kld", 0.0)),
        )

    return running / max(1, len(loader))


def _safe_id(x):
    if isinstance(x, dict) and "id" in x:
        return x["id"]
    return np.nan

def _safe_name(x):
    if isinstance(x, dict) and "name" in x:
        return x["name"]
    return None

def _safe_bool(e: dict, key: str, default=False) -> bool:
    v = e.get(key, default)
    return bool(v) if v is not None else bool(default)


def _generic_outcome_for_event(e: dict):
    """
    Returns (outcome_name, success_bool_or_nan) using StatsBomb conventions.
    - Many nested outcome fields exist; missing often implies 'Complete/Success'.
    - For event types without a notion of outcome, returns (None, np.nan).
    """
    t = _safe_name(e.get("type"))

    # PASS
    if t == "Pass" and isinstance(e.get("pass"), dict):
        out = e["pass"].get("outcome")
        if isinstance(out, dict):
            return out.get("name"), False
        # missing outcome => completed
        return "Complete", True

    # SHOT
    if t == "Shot" and isinstance(e.get("shot"), dict):
        out = e["shot"].get("outcome")
        if isinstance(out, dict):
            # success is ambiguous (goal vs on target etc). keep np.nan for boolean.
            return out.get("name"), np.nan
        return None, np.nan

    # DRIBBLE
    if t == "Dribble" and isinstance(e.get("dribble"), dict):
        out = e["dribble"].get("outcome")
        if isinstance(out, dict):
            name = out.get("name")
            # StatsBomb uses "Complete"/"Incomplete"
            if name is not None:
                return name, (name.lower() == "complete")
        return None, np.nan

    # DUEL
    if t == "Duel" and isinstance(e.get("duel"), dict):
        out = e["duel"].get("outcome")
        if isinstance(out, dict):
            name = out.get("name")
            # Often "Won"/"Lost"/"Success In Play"/etc
            if name is not None:
                low = name.lower()
                if low in ("won", "success", "success in play", "success out"):
                    return name, True
                if low in ("lost", "failure"):
                    return name, False
            return name, np.nan
        return None, np.nan

    # INTERCEPTION
    if t == "Interception" and isinstance(e.get("interception"), dict):
        out = e["interception"].get("outcome")
        if isinstance(out, dict):
            name = out.get("name")
            if name is not None:
                low = name.lower()
                if low in ("won", "success"):
                    return name, True
                if low in ("lost", "failure"):
                    return name, False
            return name, np.nan
        # If no outcome, treat as success-ish
        return "Won", True

    # BALL RECOVERY
    if t == "Ball Recovery" and isinstance(e.get("ball_recovery"), dict):
        fail = e["ball_recovery"].get("recovery_failure")
        if fail is True:
            return "Failure", False
        if fail is False:
            return "Success", True
        return None, np.nan

    # MISCONTROL (always “bad touch” in spirit)
    if t == "Miscontrol":
        return "Miscontrol", False

    # CLEARANCE (no explicit outcome)
    if t == "Clearance":
        return None, np.nan

    # PRESSURE (no explicit success)
    if t == "Pressure":
        return None, np.nan

    # FOUL COMMITTED / WON are separate event types (no outcome field)
    if t in ("Foul Committed", "Foul Won"):
        return None, np.nan

    # DEFAULT
    return None, np.nan

def flatten_events_for_match(sb_data_root: Path, match_row: dict) -> pd.DataFrame:
    match_id = match_row["match_id"]
    p = sb_data_root / "events" / f"{match_id}.json"
    ev = json.loads(p.read_text(encoding="utf-8"))

    rows = []
    for e in ev:
        loc = e.get("location", None)
        x = loc[0] if isinstance(loc, list) and len(loc) >= 2 else np.nan
        y = loc[1] if isinstance(loc, list) and len(loc) >= 2 else np.nan

        # end locations (pass/carry/shot)
        endx = endy = np.nan
        pass_length = np.nan
        pass_subtype = None

        # extra pass fields (lightweight, useful later)
        pass_height = None
        pass_cross = False
        pass_body_part = None
        pass_outcome = None
        pass_recipient_id = np.nan
        pass_recipient_name = None

        # carry distance (computed)
        carry_length = np.nan

        # shot extras
        shot_endx = shot_endy = np.nan
        shot_outcome = None
        shot_xg = np.nan
        shot_body_part = None
        shot_type = None

        # duel subtype
        duel_type = None
        duel_outcome = None

        # generic outcome
        generic_outcome, success = _generic_outcome_for_event(e)

            # time handling (robust: use minute/second)
        ts = e.get("timestamp", None)
        period = e.get("period", np.nan)

        minute = e.get("minute", np.nan)
        second = e.get("second", np.nan)

        if pd.notna(minute) and pd.notna(second):
            t_abs = float(minute) * 60.0 + float(second)
        else:
            t_abs = np.nan

        # keep name for compatibility; it's "match clock seconds"
        t_in_period = t_abs


        # pass/carry details
        if isinstance(e.get("pass"), dict):
            pe = e["pass"]
            end = pe.get("end_location", None)
            if isinstance(end, list) and len(end) >= 2:
                endx, endy = end[0], end[1]
            pass_length = pe.get("length", np.nan)
            pass_subtype = _safe_name(pe.get("type"))
            pass_height = _safe_name(pe.get("height"))
            pass_cross = bool(pe.get("cross", False))
            pass_body_part = _safe_name(pe.get("body_part"))

            out = pe.get("outcome")
            pass_outcome = _safe_name(out) if isinstance(out, dict) else None

            rec = pe.get("recipient")
            pass_recipient_id = _safe_id(rec)
            pass_recipient_name = _safe_name(rec)

        elif isinstance(e.get("carry"), dict):
            ce = e["carry"]
            end = ce.get("end_location", None)
            if isinstance(end, list) and len(end) >= 2:
                endx, endy = end[0], end[1]
            # compute carry length if we have both points
            if not (np.isnan(x) or np.isnan(y) or np.isnan(endx) or np.isnan(endy)):
                carry_length = float(np.hypot(endx - x, endy - y))

        # shot details
        if isinstance(e.get("shot"), dict):
            se = e["shot"]
            out = se.get("outcome")
            shot_outcome = _safe_name(out) if isinstance(out, dict) else None

            end = se.get("end_location", None)
            if isinstance(end, list) and len(end) >= 2:
                shot_endx, shot_endy = end[0], end[1]

            # StatsBomb xG field in open data
            shot_xg = se.get("statsbomb_xg", np.nan)

            shot_body_part = _safe_name(se.get("body_part"))
            shot_type = _safe_name(se.get("type"))

        # duel details
        if isinstance(e.get("duel"), dict):
            de = e["duel"]
            duel_type = _safe_name(de.get("type"))
            duel_outcome = _safe_name(de.get("outcome")) if isinstance(de.get("outcome"), dict) else None

        rows.append({
            "match_id": match_id,
            "competition_id": match_row["competition"]["competition_id"] if isinstance(match_row.get("competition"), dict) else match_row.get("competition_id"),
            "season_id": match_row["season"]["season_id"] if isinstance(match_row.get("season"), dict) else match_row.get("season_id"),
            "competition_name": match_row.get("competition", {}).get("competition_name", None) if isinstance(match_row.get("competition"), dict) else None,
            "season_name": match_row.get("season", {}).get("season_name", None) if isinstance(match_row.get("season"), dict) else None,

            # NEW: stable event keys
            "event_id": e.get("id", None),
            "event_index": e.get("index", np.nan),

            "type": _safe_name(e.get("type")),
            "play_pattern": _safe_name(e.get("play_pattern")),

            # NEW: player
            "player_id": _safe_id(e.get("player")),
            "player_name": _safe_name(e.get("player")),

            "team_id": _safe_id(e.get("team")),
            "team_name": _safe_name(e.get("team")),
            "possession": e.get("possession", np.nan),
            "possession_team_id": _safe_id(e.get("possession_team")),
            "possession_team_name": _safe_name(e.get("possession_team")),

            "minute": e.get("minute", np.nan),
            "second": e.get("second", np.nan),
            "timestamp": ts,
            "duration": e.get("duration", np.nan),
            "period": period,

            # NEW: convenient absolute time in seconds (for dt bins)
            "t_in_period_sec": t_in_period,
            "t_abs_sec": t_abs,

            # locations
            "x": x, "y": y,
            "endx": endx, "endy": endy,

            # pass
            "pass_length": pass_length,
            "pass_subtype": pass_subtype,
            "pass_height": pass_height,
            "pass_cross": pass_cross,
            "pass_body_part": pass_body_part,
            "pass_outcome": pass_outcome,
            "pass_recipient_id": pass_recipient_id,
            "pass_recipient_name": pass_recipient_name,

            # carry
            "carry_length": carry_length,

            # shot
            "shot_endx": shot_endx,
            "shot_endy": shot_endy,
            "shot_outcome": shot_outcome,
            "shot_xg": shot_xg,
            "shot_body_part": shot_body_part,
            "shot_type": shot_type,

            # duel
            "duel_type": duel_type,
            "duel_outcome": duel_outcome,

            # NEW: pressure flags (event-level)
            "under_pressure": _safe_bool(e, "under_pressure", False),
            "counterpress": _safe_bool(e, "counterpress", False),

            # NEW: generic outcome for CVAE v1
            "outcome": generic_outcome,
            "success": success,
        })


    df = pd.DataFrame(rows)

    is_move = df["type"].isin(["Pass", "Carry"])
    has_xy  = df[["x","y","endx","endy"]].notna().all(axis=1)

    df["dx"] = 0.0
    df["dy"] = 0.0
    m = is_move & has_xy
    df.loc[m, "dx"] = (df.loc[m, "endx"] - df.loc[m, "x"]).astype(np.float32)
    df.loc[m, "dy"] = (df.loc[m, "endy"] - df.loc[m, "y"]).astype(np.float32)

    return df

def load_competitions(sb_data_root: Path) -> pd.DataFrame:
    comp_path = sb_data_root / "competitions.json"
    comps = json.loads(comp_path.read_text(encoding="utf-8"))
    return pd.DataFrame(comps)



def pick_competitions_1516(comps, premonly = True):
    if premonly:
        TARGET = [
        ("England", "Premier League"),
    ]
    else:
        TARGET = [
        #("England", "Premier League"),
        ("Spain", "La Liga"),
        #("Italy", "Serie A"),
        #("Germany", "1. Bundesliga"),
    ]
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
def load_all_events_1516(sb_data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comps = load_competitions(sb_data_root)
    picked = pick_competitions_1516(comps, premonly=True)

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

def zone_bounds(zone_name: str):
    poly = functions.zone_polygon(zone_name)
    if poly is None:
        return None
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)

"""def xy_to_u_in_zone(x, y, zone_name: str):
    b = zone_bounds(zone_name)
    if b is None:
        return 0.0, 0.0
    x0, x1, y0, y1 = b
    ux = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    uy = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    # clamp to [0,1] in case of boundary noise
    ux = float(np.clip(ux, 0.0, 1.0))
    uy = float(np.clip(uy, 0.0, 1.0))
    return ux, uy"""




def v1_outcome(row):
    t = row["type"]
    if t == "Pass":
        if pd.isna(row["success"]): return "NA"
        return "Complete" if bool(row["success"]) else "Incomplete"
    if t == "Dribble":
        if pd.isna(row["success"]): return "NA"
        return "Complete" if bool(row["success"]) else "Incomplete"
    if t == "Shot":
        return row["shot_outcome"] if row["shot_outcome"] is not None else "NA"
    return "NA"


def terminal_reason(g):
    last = g.iloc[-1]
    t = last["type"]

    # 1) shots
    if t == "Shot":
        return "shot"

    # 2) fouls
    if t in ("Foul Committed", "Foul Won"):
        return "foul"

    # 3) ball out of play (StatsBomb has a general `out` boolean) :contentReference[oaicite:1]{index=1}
    if bool(last.get("out", False)):
        return "out"

    # 4) clear turnover-ish endings
    if t in ("Dispossessed", "Miscontrol", "Interception", "Ball Recovery", "Duel", "Block", "Clearance"):
        return "ball_loss"

    # 5) pass/dribble failure
    if t in ("Pass", "Dribble") and last.get("outcome_v1") in ("Incomplete",):
        return "turnover"

    return "other"

def build_possession_sequences(df, max_T=40):
    sequences = []
    meta = []  # store keys for matching condition vectors later
    INTERVENTION_TYPES = {
    "Duel", "Interception", "Block", "Ball Recovery",
    "Clearance", "Dispossessed", "Miscontrol",
    "Foul Won", "Foul Committed", "Pressure", "50/50"
}

    NEW_POSSESSION_TYPES = {"Pass", "Carry", "Shot"}

    grp_cols = ["match_id", "possession", "possession_team_id"]
    seq_id = 0
    for (mid, poss, ptid), g in df.groupby(grp_cols, sort=False):
        current_ptid = ptid
        g = g.sort_values(["minute","second","event_index"])
        #g = g[g["team_id"] == g["possession_team_id"]]
        if g.empty:
            continue


        steps = []
        
        seq_start_zone = None
        seq_start_minute = None
        seq_start_x = None
        seq_start_y = None
        seq_start_period = None
        for _, r in g.iterrows():

            # NEW: split sequence when opponent starts real possession
            if len(steps) > 0 and r["team_id"] != current_ptid and r["type"] in NEW_POSSESSION_TYPES:
                
                # finish current sequence
                end_reason = "turnover"   # or keep your logic if you want
                steps.append({
                    "role": "PAD",
                    "type": "END",
                    "sz": "PAD",
                    "ez": "PAD",
                    "out": "PAD",
                    "dt": "PAD",
                    "dt_sec": 0.0,
                    "term": end_reason,
                    "xy0": [0.0, 0.0],
                    "xy1": [0.0, 0.0],
                    "dxy": [0.0, 0.0],
                })

                if len(steps) > max_T:
                    steps = steps[:max_T-1] + [steps[-1]]

                if len(steps) > 1:
                    sequences.append(steps)
                    meta.append({
                    "seq_id": seq_id,
                    "match_id": mid,
                    "possession": poss,
                    "possession_team_id": current_ptid,
                    "seq_start_zone": seq_start_zone,
                    "seq_start_minute": seq_start_minute,
                    "seq_start_x": seq_start_x,
                    "seq_start_y": seq_start_y,
                    "seq_start_period": seq_start_period,
                })
                    seq_id += 1

                # start new sequence
                steps = []
                seq_start_zone = None
                seq_start_minute = None
                seq_start_x = None
                seq_start_y = None
                seq_start_period = None
                current_ptid = r["team_id"]  # update possession team
            role = "ATT" if r["team_id"] == current_ptid else "DEF"
            if "x" in r and "y" in r:
                x0, y0 = float(r["x"]), float(r["y"])
            else:
                loc = r.get("location", None)
                x0, y0 = (float(loc[0]), float(loc[1])) if (isinstance(loc, (list, tuple)) and len(loc) >= 2) else (0.0, 0.0)

            if pd.notna(r.get("endx")) and pd.notna(r.get("endy")):
                x1, y1 = float(r["endx"]), float(r["endy"])
            else:
                eloc = r.get("end_location", None)
                if isinstance(eloc, (list, tuple)) and len(eloc) >= 2:
                    x1, y1 = float(eloc[0]), float(eloc[1])
                else:
                    # if no end_location, keep equal start (or 0,0) — but we'll mask these in loss anyway
                    x1, y1 = x0, y0
   

            dx, dy = (x1 - x0), (y1 - y0)
            if len(steps) == 0:
                seq_start_zone = r["start_zone"]
                seq_start_minute = r["minute"]
                seq_start_x = x0
                seq_start_y = y0
                seq_start_period = r["period"]
            steps.append({
                "role": role,   
                "type": r["type"],
                "sz": r["start_zone"],
                "ez": r["end_zone"],
                "out": r["outcome_v1"],
                "dt": r["dt_bin"],
                "term": "NA_TERM",  # only used for END token
                "dt_sec": float(r["dt"]) if pd.notna(r["dt"]) else 0.0,
                
                              
                "xy0": [x0, y0],
                "xy1": [x1, y1],
                "dxy": [dx, dy],
            })

        # append END

        end_reason = terminal_reason(g)  
        steps.append({
            "role": "PAD",
            "type": "END",
            "sz": "PAD",
            "ez": "PAD",
            "out": "PAD",
            "dt": "PAD",
            "term": end_reason,
            "dt_sec": 0.0,
            
            "xy0": [0.0, 0.0],
            "xy1": [0.0, 0.0],
            "dxy": [0.0, 0.0],   # NEW
        })

        # truncate/pad later; but keep reasonable max length now
        if len(steps) > max_T:
            steps = steps[:max_T-1] + [steps[-1]]  # keep END
        if len(steps) == 0:
            continue

        sequences.append(steps)
        meta.append({
            "seq_id": seq_id,
            "match_id": mid,
            "possession": poss,
            "possession_team_id": current_ptid,
            "seq_start_zone": seq_start_zone,
            "seq_start_minute": seq_start_minute,
            "seq_start_x": seq_start_x,
            "seq_start_y": seq_start_y,
            "seq_start_period": seq_start_period,
        })
        seq_id += 1

    return sequences, pd.DataFrame(meta)
def build_vocab(values, add_pad=True):
    # PAD must be 0
    uniq = sorted(set(values))
    vocab = {"PAD": 0} if add_pad else {}
    for v in uniq:
        if add_pad and v == "PAD":
            continue
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab
@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device,
    type2id,
    beta=1.0,
    lambda_dxy=1.0,
    lambda_soft=0.1,
    lambda_zone=0.2,
    ez_bb=None,
    ez2id=None,
    out2id=None,
    valid_outcome_mask=None,
):
    model.eval()
    
    valid_outcome_mask = valid_outcome_mask.to(device)

    END_ID   = type2id["END"]
    PASS_ID  = type2id["Pass"]
    CARRY_ID = type2id["Carry"]

    ez_pad_id = ez2id.get("PAD") if isinstance(ez2id, dict) else None
    ez_na_id  = ez2id.get("NA_END") if isinstance(ez2id, dict) else None

    if ez_bb is not None:
        ez_bb = ez_bb.to(device)

    total = 0.0
    
    parts_sum = {"main": 0.0, "term": 0.0, "kld": 0.0, "dxy": 0.0, "zone": 0.0, "soft": 0.0, "align": 0.0}
    n = 0

    for batch in loader:
        # Expect: x, dxy_t, xy0_t, xy1_t, c, zone_code, min_code, mask, lengths
        if len(batch) != 9:
            raise ValueError(f"Expected batch of len 9, got {len(batch)}.")

        x, dxy_t, xy0_t, xy1_t, c, zone_code, min_code, mask, lengths = batch

        x         = x.to(device)
        dxy_t     = dxy_t.to(device)     # can stay unused
        xy0_t     = xy0_t.to(device)
        xy1_t     = xy1_t.to(device)
        c         = c.to(device)
        zone_code = zone_code.to(device)
        min_code  = min_code.to(device)
        mask      = mask.to(device)
        lengths   = lengths.to(device)

     
        logits, mu, logv = model(x, c, zone_code, min_code, xy0_t, xy1_t, lengths)

        loss, parts = compute_loss(
            logits=logits,
            x=x,
            dxy_t=dxy_t,
            xy0_t=xy0_t,
            xy1_t=xy1_t,
            mask=mask,
            type_end_id=END_ID,
            type_pass_id=PASS_ID,
            type_carry_id=CARRY_ID,
            mu=mu,
            logv=logv,
            beta=beta,
            lambda_dxy=lambda_dxy,
            lambda_soft=lambda_soft,
            lambda_zone=lambda_zone,
            ez_bb=ez_bb,
            ez_pad_id=ez_pad_id,
            ez_na_id=ez_na_id,
            valid_outcome_mask=valid_outcome_mask
        )

        total += float(loss.item())
        for k in parts_sum.keys():
            parts_sum[k] += float(parts.get(k, 0.0))
        n += 1

    out = {k: v / max(1, n) for k, v in parts_sum.items()}
    out["total"] = total / max(1, n)
    return out
def zone_bbox(zone: str):
    """Return (x0,x1,y0,y1) bbox in StatsBomb coords for a zone name."""
    if zone in (None, "NA_END", "PAD"):
        return None

    # penalty boxes
    if zone.startswith("PenBox_Def_"):
        x0, x1 = functions.PA_L_X0, functions.PA_L_X1
        if zone.endswith("_Left"):
            y0, y1 = functions.PA_Y0, functions.PA_SPLIT_Y1
        elif zone.endswith("_Central"):
            y0, y1 = functions.PA_SPLIT_Y1, functions.PA_SPLIT_Y2
        else:  # _Right
            y0, y1 = functions.PA_SPLIT_Y2, functions.PA_Y1
        return (x0, x1, y0, y1)

    if zone.startswith("PenBox_Att_"):
        x0, x1 = functions.PA_R_X0, functions.PA_R_X1
        if zone.endswith("_Left"):
            y0, y1 = functions.PA_Y0, functions.PA_SPLIT_Y1
        elif zone.endswith("_Central"):
            y0, y1 = functions.PA_SPLIT_Y1, functions.PA_SPLIT_Y2
        else:
            y0, y1 = functions.PA_SPLIT_Y2, functions.PA_Y1
        return (x0, x1, y0, y1)

    # center dead
    if zone == "Center_Dead_Def":
        return (functions.MID_DEF_X0, functions.MID_DEF_X1, functions.MID_Y0, functions.MID_Y1)
    if zone == "Center_Dead_Att":
        return (functions.MID_ATT_X0, functions.MID_ATT_X1, functions.MID_Y0, functions.MID_Y1)

    # wings
    m = re.match(r"^Wing_Left_Zone(\d+)$", zone)
    if m:
        k = int(m.group(1))
        x0, x1 = functions.x_bins_wide[k], functions.x_bins_wide[k+1]
        return (float(x0), float(x1), 0.0, float(functions.Y_TOP))

    m = re.match(r"^Wing_Right_Zone(\d+)$", zone)
    if m:
        k = int(m.group(1))
        x0, x1 = functions.x_bins_wide[k], functions.x_bins_wide[k+1]
        return (float(x0), float(x1), float(functions.Y_BOT), 80.0)

    # pockets
    m = re.match(r"^(Def|Att)_Pocket_(Left|Central|Right)$", zone)
    if m:
        side, band = m.group(1), m.group(2)
        x0, x1 = (0.0, 60.0) if side == "Def" else (60.0, 120.0)

        if band == "Left":
            y0, y1 = float(functions.Y_TOP), 30.0
        elif band == "Central":
            y0, y1 = 30.0, 50.0
        else:
            y0, y1 = 50.0, float(functions.Y_BOT)

        return (x0, x1, y0, y1)

    # unknown label
    return None

def zone_u_to_point(zone: str, u, fallback=None):
    """
    zone: zone name
    u: (ux,uy) in [0,1]
    returns (x,y) in StatsBomb coords
    """
    bb = zone_bbox(zone)
    if bb is None:
        return fallback
    x0, x1, y0, y1 = bb
    ux, uy = u
    ux = float(np.clip(ux, 0.0, 1.0))
    uy = float(np.clip(uy, 0.0, 1.0))
    x = x0 + ux * (x1 - x0)
    y = y0 + uy * (y1 - y0)
    return (float(x), float(y))

import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Beta



@torch.no_grad()
def sb_dxy_to_meters(dxdy):
    mx = 105.0 / 120.0
    my = 68.0 / 80.0
    out = dxdy.clone()
    out[..., 0] = out[..., 0] * mx
    out[..., 1] = out[..., 1] * my
    return out


###NOT UPDATED@torch.no_grad()
def validate_dxy_meters(
    model,
    loader,
    device,
    type2id,
    max_batches=None,
):
    model.eval()
    PASS_ID  = type2id["Pass"]
    CARRY_ID = type2id["Carry"]

    dists = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        # Expect: x, dxy_t, xy0_t, xy1_t, c, zone_code, min_code, mask, lengths
        if len(batch) != 9:
            raise ValueError(f"Expected batch of len 9, got {len(batch)}.")

        x, dxy_t, xy0_t, xy1_t, c, zone_code, min_code, mask, lengths = batch

        x         = x.to(device)
        xy0_t     = xy0_t.to(device)
        xy1_t     = xy1_t.to(device)
        c         = c.to(device)
        zone_code = zone_code.to(device)
        min_code  = min_code.to(device)
        mask      = mask.to(device)
        lengths   = lengths.to(device)

      
        logits, mu, logv = model(x, c, zone_code, min_code, xy0_t, xy1_t, lengths)
        # dxy_true returned may contain NaNs (good)

        role_t, type_t, sz_t, ez_t, out_t, dt_t, term_t = x.unbind(dim=-1)

        is_move = (type_t == PASS_ID) | (type_t == CARRY_ID)
        coord_ok = torch.isfinite(xy0_t).all(dim=-1) & torch.isfinite(xy1_t).all(dim=-1)
        m = is_move & mask.bool() & coord_ok

        if m.sum().item() == 0:
            continue
        pred = logits["dxy"][m]          
        true = (xy1_t - xy0_t)[m]        

        diff = pred - true   

        dx_m = diff[:, 0] 
        dy_m = diff[:, 1] 
        dist = torch.sqrt(dx_m**2 + dy_m**2)
        dists.append(dist.detach().cpu())

    if len(dists) == 0:
        return {"mean_m": float("nan"), "median_m": float("nan"), "n": 0}

    d = torch.cat(dists, dim=0).numpy().astype(np.float32)
    return {"mean_m": float(d.mean()), "median_m": float(np.median(d)), "n": int(d.size)}