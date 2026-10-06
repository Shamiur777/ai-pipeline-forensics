"""Log miner: production LLM logs (JSONL) -> labelled eval cases, incrementally.

Pipeline per run: skip already-seen log ids -> explicit signals -> TF-IDF clustering for
rarity -> diverse representatives per cluster -> dedupe (in-run and against the dataset)
-> stratify across failure types under a per-run budget -> write cases + update state.

Log record: {id, ts, model, prompt_version, request, response, latency_ms, error,
             feedback: {thumbs: "up"|"down"|null, edited_response: str|null}, retry: bool}
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import EvalStore, input_hash

SIGNAL_WEIGHT = {"error": 3.0, "user_edit": 3.0, "thumbs_down": 2.5, "retry": 2.0, "refusal": 2.0,
                 "latency_outlier": 1.5, "rare_cluster": 1.5, "length_anomaly": 1.0}
SEVERITY = {"error": "high", "user_edit": "high", "thumbs_down": "high", "retry": "medium", "refusal": "medium",
            "latency_outlier": "low", "rare_cluster": "low", "length_anomaly": "low"}
REFUSAL_RE = re.compile(r"\b(i can'?t (help|assist)|i cannot (help|assist)|i'?m (sorry|unable)|unable to (help|assist)"
                        r"|as an ai)", re.I)
STOP = set("the a an of to for in on and or is are my i me how do what can you it with this that be".split())


@dataclass
class MinerConfig:
    budget: int = 60            # max new cases per run
    per_cluster: int = 3        # diverse representatives kept per (cluster, failure type)
    sim_threshold: float = 0.45  # cosine similarity to join a cluster
    max_share: float = 0.4      # per-type cap (fraction of budget), applied when >= 3 types compete
    latency_k: float = 4.0      # robust z-score for latency outliers


@dataclass
class RunReport:
    run: int
    new_logs: int
    flagged: int
    dups_removed: int
    selected: int
    added: int
    by_type: dict = field(default_factory=dict)
    dataset_size: int = 0


def load_logs(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------- explicit signals ----------

def _words(text: str) -> int:
    return len(text.split())


def batch_stats(logs: list[dict]) -> dict:
    by_model = defaultdict(list)
    for l in logs:
        if l.get("latency_ms") is not None:
            by_model[l.get("model")].append(l["latency_ms"])
    lat = {}
    for model, vals in by_model.items():
        med = statistics.median(vals)
        mad = statistics.median(abs(v - med) for v in vals) or 1.0
        lat[model] = (med, 1.4826 * mad)
    lens = [_words(l.get("response") or "") for l in logs if l.get("response")]
    return {"latency": lat, "median_words": statistics.median(lens) if lens else 0}


def detect_signals(log: dict, stats: dict, cfg: MinerConfig | None = None) -> list[str]:
    cfg = cfg or MinerConfig()
    out, fb = [], log.get("feedback") or {}
    resp = log.get("response") or ""
    if log.get("error"):
        out.append("error")
    if fb.get("thumbs") == "down":
        out.append("thumbs_down")
    if fb.get("edited_response") and fb["edited_response"].strip() != resp.strip():
        out.append("user_edit")
    if log.get("retry"):
        out.append("retry")
    if resp and REFUSAL_RE.search(resp):
        out.append("refusal")
    med, sigma = stats["latency"].get(log.get("model"), (None, None))
    lat = log.get("latency_ms")
    if lat is not None and med is not None and lat > max(med + cfg.latency_k * sigma, 3 * med):
        out.append("latency_outlier")
    n, med_words = _words(resp), stats["median_words"]
    if resp and ((med_words and n >= 5 * med_words) or (n <= 2 and _words(log.get("request") or "") >= 4)):
        out.append("length_anomaly")
    return out


# ---------- TF-IDF clustering (pure Python) ----------

def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    docs = [Counter(_tokens(t)) for t in texts]
    df = Counter(tok for d in docs for tok in d)
    n = len(texts)
    vecs = []
    for d in docs:
        v = {tok: (1 + math.log(c)) * (math.log((1 + n) / (1 + df[tok])) + 1) for tok, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({k: x / norm for k, x in v.items()})
    return vecs


def cosine(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(k, 0.0) for k, x in a.items())


def cluster(vecs: list[dict], threshold: float) -> list[int]:
    """Leader clustering: each item joins the most similar leader, or starts a new cluster."""
    leaders: list[int] = []
    assign = []
    for i, v in enumerate(vecs):
        best, best_sim = -1, 0.0
        for cid, li in enumerate(leaders):
            s = cosine(v, vecs[li])
            if s > best_sim:
                best, best_sim = cid, s
        if best >= 0 and best_sim >= threshold:
            assign.append(best)
        else:
            leaders.append(i)
            assign.append(len(leaders) - 1)
    return assign


def diverse_pick(idxs: list[int], vecs: list[dict], scores: dict[int, float], k: int) -> list[int]:
    """Greedy farthest-point selection, seeded by the highest-scoring item."""
    if len(idxs) <= k:
        return list(idxs)
    chosen = [max(idxs, key=lambda i: (scores[i], -i))]
    while len(chosen) < k:
        rest = [i for i in idxs if i not in chosen]
        chosen.append(max(rest, key=lambda i: (min(1 - cosine(vecs[i], vecs[c]) for c in chosen), scores[i], -i)))
    return chosen


# ---------- state + run ----------

def load_state(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"seen_ids": [], "runs": []}


def mine(log_path: str | Path, store: EvalStore, state_path: str | Path, cfg: MinerConfig | None = None) -> RunReport:
    cfg = cfg or MinerConfig()
    state = load_state(state_path)
    seen = set(state["seen_ids"])
    new = sorted((l for l in load_logs(log_path) if l["id"] not in seen), key=lambda l: l["id"])
    run_no = len(state["runs"]) + 1
    if not new:
        rep = RunReport(run_no, 0, 0, 0, 0, 0, {}, len(store))
        state["runs"].append(rep.__dict__)
        Path(state_path).write_text(json.dumps(state, indent=1), encoding="utf-8")
        return rep

    stats = batch_stats(new)
    vecs = tfidf_vectors([l.get("request") or "" for l in new])
    assign = cluster(vecs, cfg.sim_threshold)
    sizes = Counter(assign)
    rare_max = max(3, math.ceil(0.02 * len(new)))

    signals, scores = {}, {}
    for i, log in enumerate(new):
        sig = detect_signals(log, stats, cfg)
        if sizes[assign[i]] <= rare_max:
            sig.append("rare_cluster")
        if sig:
            signals[i] = sig
            scores[i] = sum(SIGNAL_WEIGHT[s] for s in sig) + 1.5 / math.sqrt(sizes[assign[i]])
    flagged = len(signals)

    def primary(i):
        return max(signals[i], key=lambda s: SIGNAL_WEIGHT[s])

    # diverse representatives per (cluster, failure type), so one noisy type cannot crowd out another
    by_cluster = defaultdict(list)
    for i in signals:
        by_cluster[(assign[i], primary(i))].append(i)
    kept = [i for idxs in by_cluster.values() for i in diverse_pick(idxs, vecs, scores, cfg.per_cluster)]

    # dedupe: against the dataset, then within the run (keep the best-scoring copy)
    best: dict[str, int] = {}
    dups = 0
    for i in sorted(kept, key=lambda i: (-scores[i], i)):
        h = input_hash(new[i].get("request") or "")
        if store.has(new[i].get("request") or "") or h in best:
            dups += 1
        else:
            best[h] = i
    candidates = list(best.values())

    # stratify: round-robin over primary failure types, best score first
    pools = defaultdict(list)
    for i in sorted(candidates, key=lambda i: (-scores[i], i)):
        pools[primary(i)].append(i)
    cap = math.ceil(cfg.max_share * cfg.budget) if len(pools) >= 3 else cfg.budget
    taken: Counter = Counter()
    selected: list[int] = []
    while len(selected) < cfg.budget and any(pools[t] and taken[t] < cap for t in pools):
        for t in sorted(pools, key=lambda t: (taken[t], t)):
            if pools[t] and taken[t] < cap and len(selected) < cfg.budget:
                selected.append(pools[t].pop(0))
                taken[t] += 1

    added = 0
    for i in selected:
        log, sig = new[i], signals[i]
        edit = (log.get("feedback") or {}).get("edited_response")
        ptype = primary(i)
        expected = edit.strip() if "user_edit" in sig and edit else None
        tags = [f"type:{ptype}", f"severity:{SEVERITY[ptype]}", "source:prod_log", f"model:{log.get('model')}",
                f"prompt_version:{log.get('prompt_version')}", *[f"signal:{s}" for s in sig]]
        meta = {"log_id": log["id"], "run": run_no, "signals": sig, "score": round(scores[i], 3),
                "cluster_id": assign[i], "cluster_size": sizes[assign[i]], "original_response": log.get("response"),
                "ts": log.get("ts")}
        weight = 1.0 + min(2.0, 1.0 / math.sqrt(sizes[assign[i]]) * 2)  # rarer interactions weigh more
        if store.add(log["request"], expected, tags, round(weight, 2), meta,
                     needs_review_reason=f"{ptype}: no correction available"):
            added += 1

    rep = RunReport(run_no, len(new), flagged, dups, len(selected), added, dict(taken), len(store))
    state["seen_ids"] = sorted(seen | {l["id"] for l in new})
    state["runs"].append(rep.__dict__)
    Path(state_path).write_text(json.dumps(state, indent=1), encoding="utf-8")
    return rep
