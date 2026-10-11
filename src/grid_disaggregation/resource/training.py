"""Masked three-seed training with validation-only checkpoint selection."""
from pathlib import Path
import json
import random
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score
from . import data as res
from ..common import dump, sha, load_plan
SEEDS = [42, 2026, 3407]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_bce(logits, target, mask, pos_weight):
    elem = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    count = mask.sum(0)
    return ((elem * mask).sum(0) / count.clamp_min(1))[count > 0].mean()


def score(y, mask, prob, thresholds):
    rows = res.metrics(y, mask, prob, thresholds)
    aps = [r["pr_auc_ap"] for r in rows if r["pr_auc_ap"] is not None]
    return {
        "macro_pr_auc": float(np.mean(aps)) if aps else None,
        "macro_f1": float(np.mean([r["f1"] for r in rows])) if rows else None,
        "PV_f1": next((r["f1"] for r in rows if r["label"] == "pv_active"), None),
        "ESS_f1": next((r["f1"] for r in rows if r["label"] == "energy_above_5kw"), None),
        "EV_f1": next((r["f1"] for r in rows if r["label"] == "charger_above_1kw"), None),
    }, rows


def train_seed(seed, train, val, aux, weights, out, epochs=30, patience=6):
    seed_all(seed)
    model = res.TCNLSTMClassifier(res.ModelConfig(input_channels=114)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 5000)
    logs = []
    start = time.perf_counter()

    # EV-only auxiliary pretraining uses independent raw charger records.  The
    # first two labels stay masked, so these records cannot become fake PV/ESS labels.
    for epoch in range(1, 4) if aux is not None else []:
        model.train(); losses = []
        order = rng.permutation(aux["ends"])
        for j in range(0, len(order), 256):
            x, y, m = res.get_batch(aux, order[j:j + 256])
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(x), y, m, weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({"seed": seed, "stage": "ev_auxiliary_pretrain", "epoch": epoch, "loss": float(np.mean(losses))})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    vy, vm = res.truth(val)
    best, stale, best_epoch = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        order = rng.permutation(train["ends"])
        for j in range(0, len(order), 256):
            x, y, m = res.get_batch(train, order[j:j + 256])
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(x), y, m, weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        prob = res.predict(model, val)
        metrics, _ = score(vy, vm, prob, [.5, .5, .5])
        ap = metrics["macro_pr_auc"]
        logs.append({"seed": seed, "stage": "main_raw_finetune", "epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if ap > best + 1e-5:
            best, stale, best_epoch = ap, 0, epoch
            torch.save({"state_dict": model.state_dict(), "config": res.ModelConfig(input_channels=114).to_dict(),
                        "task": "resource", "architecture": "tcn_lstm", "seed": seed}, out / f"seed_{seed}_best_validation.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    pd.DataFrame(logs).to_csv(out / f"seed_{seed}_epochs.csv", index=False)
    model.load_state_dict(torch.load(out / f"seed_{seed}_best_validation.pt", map_location=DEVICE, weights_only=True)["state_dict"])
    prob = res.predict(model, val)
    thresholds = []
    for k in range(3):
        ok = vm[:, k]; grid = np.linspace(.05, .95, 37)
        thresholds.append(float(grid[np.argmax([f1_score(vy[ok, k], prob[ok, k] >= t, zero_division=0) for t in grid])]))
    final, _ = score(vy, vm, prob, thresholds)
    return {"seed": seed, "best_epoch": best_epoch, "validation_thresholds": thresholds,
            "checkpoint_sha256": sha(out / f"seed_{seed}_best_validation.pt"),
            "seconds": time.perf_counter() - start, **final}, model


def evaluate(segments, results, models, out):
    rows = []
    frames = []
    for split in ["validation", "internal_test", "domain_test"]:
        if not any(s['split']==split and s['mask'].any() for s in segments):
            continue
        pack = res.pack(segments, split, DEVICE)
        y, mask = res.truth(pack)
        frame = pd.DataFrame(pack["meta"], columns=["batch", "minute_start", "main_power_kw"])
        frame["split"] = split
        for result, model in zip(results, models):
            p = res.predict(model, pack)
            met, detail = score(y, mask, p, result["validation_thresholds"])
            rows.extend({"split": split, "seed": result["seed"], **r} for r in detail)
            for k, label in enumerate(res.LABELS):
                frame[f"{label}_prob_seed{result['seed']}"] = p[:, k]
        frames.append(frame)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(frames, ignore_index=True).to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")

def run(plan_path, fields_path, output):
    torch.set_num_threads(4)
    out = Path(output)
    (out/'checkpoints').mkdir(parents=True, exist_ok=True)
    plan = load_plan(plan_path)
    fields = json.loads(Path(fields_path).read_text(encoding='utf-8'))['fields']
    segments, norm = res.prepare(plan, fields, out/'data')
    auxiliary = res.prepare_auxiliary(plan, norm)
    train = res.pack(segments, 'train', DEVICE)
    val = res.pack(segments, 'validation', DEVICE)
    aux = res.pack(auxiliary, 'train', DEVICE) if auxiliary else None
    y, mask = res.truth(train)
    pos = (y*mask).sum(0)
    neg = ((1-y)*mask).sum(0)
    if not ((pos>0)&(neg>0)).all():
        raise ValueError('Each resource needs known positive and negative TRAIN endpoints')
    weights = torch.tensor(np.clip(neg/np.maximum(pos, 1), .25, 10), dtype=torch.float32, device=DEVICE)
    results, models = [], []
    for seed in SEEDS:
        result, model = train_seed(seed, train, val, aux, weights, out/'checkpoints')
        results.append(result)
        models.append(model.eval())
    selected = sorted(results, key=lambda r: (-r['macro_pr_auc'], -r['macro_f1'], r['seed']))[0]
    dump(out/'selection.json', {'selected_seed': selected['seed'], 'thresholds': selected['validation_thresholds'],
        'rule': 'Validation macro PR-AUC, then macro F1', 'test_used_for_selection': False})
    shutil.copy2(out/'checkpoints'/f"seed_{selected['seed']}_best_validation.pt", out/'selected.pt')
    pd.DataFrame(results).to_csv(out/'seed_summary.csv', index=False)
    evaluate(segments, results, models, out)
