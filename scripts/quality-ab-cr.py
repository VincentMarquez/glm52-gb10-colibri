#!/usr/bin/env python3
"""
CACHE_ROUTE quality A/B with live progress.

- Streams engine SCORE output as it runs (tok/s, hit%, running accuracy)
- Appends a JSONL checkpoint every N requests so Ctrl-C still leaves usable results
- Runs CACHE_ROUTE=0 then CACHE_ROUTE=1 on the same questions

Usage:
  python3 ~/quality-ab-cr.py --limit 20 --tasks hellaswag
  python3 ~/quality-ab-cr.py --limit 40 --tasks hellaswag,arc_challenge,mmlu

Stop anytime (Ctrl-C). Check:
  ~/quality-ab-cr/latest/STATUS.txt
  ~/quality-ab-cr/latest/*.progress.jsonl
  ~/quality-ab-cr/latest/SUMMARY.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------- paths / defaults ----------
SNAP = os.environ.get("COLI_MODEL", os.path.expanduser("~/glm52-colibri-int4"))
GLM = os.environ.get("GLM_BIN", os.path.expanduser("~/colibri/c/glm"))
DATA = os.environ.get("BENCH_DATA", os.path.expanduser("~/colibri/c/bench"))
OUT_ROOT = Path(os.environ.get("QUALITY_AB_DIR", os.path.expanduser("~/quality-ab-cr")))

SMOKE = [
    {"ctx": "The capital of France is", "choices": [" Paris", " Berlin", " Rome"], "gold": 0},
    {"ctx": "2 + 2 =", "choices": [" 4", " 5", " 7"], "gold": 0},
    {"ctx": "The sun rises in the", "choices": [" east", " west", " north"], "gold": 0},
]


def load_docs(task: str, data_dir: str, limit: int, seed: int):
    if task == "smoke":
        return SMOKE[:limit] if limit else list(SMOKE)
    path = os.path.join(data_dir, task + ".jsonl")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — need hellaswag/arc_challenge/mmlu jsonl")
    docs = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(docs)
    return docs[:limit] if limit else docs


def build_requests(tk, docs_by_task):
    """Same layout as tools/eval_glm.py."""
    reqs, meta, perq = [], [], {}
    tok_counts = []  # total tokens per request (ctx+cont)
    for t, docs in docs_by_task.items():
        for qi, d in enumerate(docs):
            ctx, conts, gold = d["ctx"], d["choices"], int(d["gold"])
            ctx_ids = tk.encode(ctx).ids
            for oi, cont in enumerate(conts):
                full = tk.encode(ctx + cont).ids
                cl = len(ctx_ids)
                while cl > 0 and (cl > len(full) or full[:cl] != ctx_ids[:cl]):
                    cl -= 1
                cont_ids = full[cl:]
                if not cont_ids:
                    full = ctx_ids + tk.encode(cont).ids
                    cl = len(ctx_ids)
                    cont_ids = full[cl:]
                if cl < 1:
                    cl = 1
                T = len(full)
                reqs.append(f"{cl} {T - cl} " + " ".join(map(str, full)))
                meta.append((t, qi, oi, T - cl, max(1, len(cont)), gold, T))
                perq.setdefault((t, qi), []).append(len(meta) - 1)
                tok_counts.append(T)
    return reqs, meta, perq, tok_counts


def write_status(path: Path, text: str):
    path.write_text(text)
    # also mirror last line-ish to a live STATUS for tail -f
    print(text.rstrip(), flush=True)


def running_accuracy(meta, perq, lp_partial: list[float | None]):
    """Accuracy over questions that have ALL options scored so far."""
    by_task = {}
    for (t, qi), ridx in perq.items():
        if any(lp_partial[r] is None for r in ridx):
            continue
        gold = meta[ridx[0]][5]
        best = max(ridx, key=lambda r: lp_partial[r])
        bestn = max(ridx, key=lambda r: lp_partial[r] / meta[r][4])
        d = by_task.setdefault(t, {"n": 0, "acc": 0, "accn": 0})
        d["n"] += 1
        d["acc"] += int(meta[best][2] == gold)
        d["accn"] += int(meta[bestn][2] == gold)
    return by_task


def format_acc(by_task: dict) -> str:
    if not by_task:
        return "acc: (no complete questions yet)"
    parts = []
    for t, d in sorted(by_task.items()):
        n = d["n"]
        parts.append(f"{t} n={n} acc={100*d['acc']/n:.0f}% acc_norm={100*d['accn']/n:.0f}%")
    return " | ".join(parts)


class Stopped(Exception):
    pass


def run_cell(
    *,
    label: str,
    env_extra: dict,
    reqs: list[str],
    meta: list,
    perq: dict,
    tok_counts: list[int],
    cell_dir: Path,
    glm: str,
    snap: str,
    cap: int,
    checkpoint_every: int,
):
    cell_dir.mkdir(parents=True, exist_ok=True)
    req_path = cell_dir / "requests.txt"
    req_path.write_text("\n".join(reqs) + "\n")
    progress_path = cell_dir / "progress.jsonl"
    log_path = cell_dir / "engine.stderr.log"
    out_path = cell_dir / "scores.txt"
    status_path = cell_dir / "STATUS.txt"
    summary_path = cell_dir / "SUMMARY.txt"

    # wipe progress for this cell (fresh run)
    for p in (progress_path, log_path, out_path, summary_path):
        if p.exists():
            p.unlink()

    env = dict(os.environ)
    env.update(env_extra)
    env["SNAP"] = snap
    env["SCORE"] = str(req_path)
    # keep noise down but allow hit%/route banners
    env.setdefault("MTP", "0")  # SCORE path doesn't need drafts

    cmd = [glm, str(cap)]
    banner = (
        f"\n======== {label} ========\n"
        f"start: {datetime.now(timezone.utc).isoformat()}\n"
        f"cmd: {' '.join(cmd)}\n"
        f"env: { {k: env[k] for k in sorted(env_extra)} }\n"
        f"requests: {len(reqs)}\n"
        f"progress: {progress_path}\n"
        f"stderr log: {log_path}\n"
        f"==========================\n"
    )
    write_status(status_path, banner)

    t0 = time.time()
    lp: list[float | None] = [None] * len(reqs)
    tokens_done = 0
    n_done = 0
    stopped = False

    # line-buffered child where possible
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # drain stderr in a simple non-blocking-ish way via threads
    import threading

    stderr_tail: list[str] = []

    def drain_stderr():
        assert proc.stderr is not None
        with open(log_path, "a") as lf:
            for line in proc.stderr:
                lf.write(line)
                lf.flush()
                s = line.rstrip()
                stderr_tail.append(s)
                if len(stderr_tail) > 40:
                    del stderr_tail[0]
                # surface engine progress lines live
                if s.startswith("[score") or "CACHE_ROUTE" in s or s.startswith("[CUDA") or "hit" in s.lower():
                    print(f"  [engine] {s}", flush=True)

    th = threading.Thread(target=drain_stderr, daemon=True)
    th.start()

    def handle_sig(signum, frame):
        nonlocal stopped
        stopped = True
        print(f"\n!! signal {signum} — stopping after current request, saving partial results...", flush=True)
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass

    old_int = signal.signal(signal.SIGINT, handle_sig)
    old_term = signal.signal(signal.SIGTERM, handle_sig)

    try:
        assert proc.stdout is not None
        with open(out_path, "a") as of:
            for line in proc.stdout:
                line = line.strip()
                if not line or line[0] not in "-0123456789":
                    continue
                parts = line.split()
                try:
                    val = float(parts[0])
                except ValueError:
                    continue
                if n_done >= len(reqs):
                    break
                lp[n_done] = val
                tokens_done += tok_counts[n_done]
                of.write(line + "\n")
                of.flush()

                rec = {
                    "i": n_done,
                    "lp": val,
                    "tokens": tok_counts[n_done],
                    "task": meta[n_done][0],
                    "qi": meta[n_done][1],
                    "oi": meta[n_done][2],
                    "t_wall": time.time() - t0,
                }
                with open(progress_path, "a") as pf:
                    pf.write(json.dumps(rec) + "\n")

                n_done += 1
                dt = max(1e-6, time.time() - t0)
                req_s = n_done / dt
                tok_s = tokens_done / dt
                by = running_accuracy(meta, perq, lp)
                n_q = sum(d["n"] for d in by.values())
                msg = (
                    f"[{label}] req {n_done}/{len(reqs)} ({100*n_done/len(reqs):.0f}%) | "
                    f"{req_s:.2f} req/s | ~{tok_s:.1f} tok/s | "
                    f"tokens {tokens_done} | wall {dt:.0f}s | "
                    f"complete_q={n_q} | {format_acc(by)}"
                )
                write_status(status_path, msg + "\n")

                if n_done % checkpoint_every == 0:
                    # checkpoint snapshot
                    snap_path = cell_dir / f"checkpoint_{n_done:05d}.json"
                    snap_path.write_text(
                        json.dumps(
                            {
                                "label": label,
                                "n_done": n_done,
                                "n_total": len(reqs),
                                "tokens_done": tokens_done,
                                "wall_s": dt,
                                "req_per_s": req_s,
                                "tok_per_s": tok_s,
                                "running_acc": {
                                    t: {
                                        "n": d["n"],
                                        "acc": 100 * d["acc"] / d["n"],
                                        "acc_norm": 100 * d["accn"] / d["n"],
                                    }
                                    for t, d in by.items()
                                },
                                "env": env_extra,
                            },
                            indent=2,
                        )
                    )

                if stopped:
                    break
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
        th.join(timeout=2)

    dt = max(1e-6, time.time() - t0)
    by = running_accuracy(meta, perq, lp)
    summary = {
        "label": label,
        "finished": n_done == len(reqs) and not stopped,
        "stopped_early": stopped or n_done < len(reqs),
        "n_done": n_done,
        "n_total": len(reqs),
        "tokens_done": tokens_done,
        "wall_s": dt,
        "req_per_s": n_done / dt,
        "tok_per_s": tokens_done / dt,
        "running_acc": {
            t: {
                "n": d["n"],
                "acc_pct": 100 * d["acc"] / d["n"],
                "acc_norm_pct": 100 * d["accn"] / d["n"],
            }
            for t, d in by.items()
        },
        "env": env_extra,
        "stderr_tail": stderr_tail[-10:],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    human = [
        f"=== SUMMARY {label} ===",
        f"done: {n_done}/{len(reqs)} requests ({'COMPLETE' if summary['finished'] else 'PARTIAL — stop OK'})",
        f"wall: {dt:.1f}s | {n_done/dt:.2f} req/s | ~{tokens_done/dt:.1f} tok/s",
        f"tokens processed: {tokens_done}",
        format_acc(by),
        f"files: {cell_dir}",
        "",
    ]
    write_status(status_path, "\n".join(human))
    return summary


def main():
    ap = argparse.ArgumentParser(description="CACHE_ROUTE quality A/B with live progress")
    ap.add_argument("--snap", default=SNAP)
    ap.add_argument("--glm", default=GLM)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tasks", default="hellaswag")
    ap.add_argument("--limit", type=int, default=20, help="questions per task (each has multiple options)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--ram", type=int, default=0)
    ap.add_argument("--route-j", type=int, default=2)
    ap.add_argument("--route-m", type=int, default=12)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--only", choices=("off", "on", "both"), default="both")
    ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args()

    if not os.path.isfile(a.glm):
        sys.exit(f"glm binary not found: {a.glm}")
    if not os.path.isdir(a.snap):
        sys.exit(f"model snap not found: {a.snap}")

    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(os.path.join(a.snap, "tokenizer.json"))
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    docs_by_task = {t: load_docs(t, a.data, a.limit, a.seed) for t in tasks}
    for t, d in docs_by_task.items():
        print(f"[{t}] {len(d)} questions", flush=True)

    reqs, meta, perq, tok_counts = build_requests(tk, docs_by_task)
    total_tok = sum(tok_counts)
    print(f"total requests (options): {len(reqs)} | total tokens ~{total_tok}", flush=True)
    print(f"note: SCORE mode scores log-likelihood (full forwards), not chat decode.", flush=True)
    print(f"      ~tok/s = tokens in scored sequences / wall time.", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(a.out) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = Path(a.out) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir)

    # save plan
    (run_dir / "PLAN.json").write_text(
        json.dumps(
            {
                "tasks": tasks,
                "limit": a.limit,
                "seed": a.seed,
                "n_requests": len(reqs),
                "total_tokens": total_tok,
                "snap": a.snap,
                "glm": a.glm,
                "route_j": a.route_j,
                "route_m": a.route_m,
            },
            indent=2,
        )
    )
    # human live pointer
    live = run_dir / "LIVE.txt"
    live.write_text(
        f"Run dir: {run_dir}\n"
        f"Tail live status:  tail -f {run_dir}/*/STATUS.txt\n"
        f"Or:               tail -f {run_dir}/LIVE.txt  (this file rewritten each progress tick)\n"
        f"Stop: Ctrl-C — partial SUMMARY + progress.jsonl will remain.\n"
    )
    print(f"\n>>> LIVE OUTPUT DIR: {run_dir}", flush=True)
    print(f">>> shortcut:       {latest}", flush=True)
    print(f">>> tail -f {latest}/cr_off/STATUS.txt", flush=True)
    print(f">>> stop with Ctrl-C anytime; partial results are kept.\n", flush=True)

    base_env = {}
    if a.ram:
        base_env["RAM_GB"] = str(a.ram)

    results = []
    cells = []
    if a.only in ("off", "both"):
        cells.append(
            (
                "cr_off",
                {
                    **base_env,
                    "CACHE_ROUTE": "0",
                },
            )
        )
    if a.only in ("on", "both"):
        cells.append(
            (
                "cr_on",
                {
                    **base_env,
                    "CACHE_ROUTE": "1",
                    "ROUTE_J": str(a.route_j),
                    "ROUTE_M": str(a.route_m),
                    "ROUTE_AGREE": "1",
                },
            )
        )

    for label, envx in cells:
        cell_dir = run_dir / label
        try:
            s = run_cell(
                label=label,
                env_extra=envx,
                reqs=reqs,
                meta=meta,
                perq=perq,
                tok_counts=tok_counts,
                cell_dir=cell_dir,
                glm=a.glm,
                snap=a.snap,
                cap=a.cap,
                checkpoint_every=a.checkpoint_every,
            )
            results.append(s)
            live.write_text(json.dumps(s, indent=2) + "\n")
        except KeyboardInterrupt:
            print("interrupted between cells", flush=True)
            break

    # comparison if both have any complete questions
    cmp_path = run_dir / "COMPARE.txt"
    lines = ["=== CACHE_ROUTE quality A/B ===", f"dir: {run_dir}", ""]
    for s in results:
        lines.append(
            f"{s['label']}: {s['n_done']}/{s['n_total']} req | "
            f"{s['req_per_s']:.2f} req/s | ~{s['tok_per_s']:.1f} tok/s | "
            f"{'DONE' if s['finished'] else 'PARTIAL'}"
        )
        for t, d in s.get("running_acc", {}).items():
            lines.append(f"  {t}: n={d['n']} acc={d['acc_pct']:.1f}% acc_norm={d['acc_norm_pct']:.1f}%")
        lines.append("")

    if len(results) == 2:
        a0, a1 = results[0], results[1]
        lines.append("Delta (on - off) where both have the task:")
        for t in sorted(set(a0.get("running_acc", {})) & set(a1.get("running_acc", {}))):
            d0, d1 = a0["running_acc"][t], a1["running_acc"][t]
            if d0["n"] and d1["n"]:
                lines.append(
                    f"  {t}: acc_norm {d1['acc_norm_pct'] - d0['acc_norm_pct']:+.1f} pp "
                    f"(off {d0['acc_norm_pct']:.1f}% n={d0['n']} → on {d1['acc_norm_pct']:.1f}% n={d1['n']})"
                )
    cmp_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nAll artifacts: {run_dir}", flush=True)
    print(f"Latest symlink: {latest}", flush=True)


if __name__ == "__main__":
    main()
