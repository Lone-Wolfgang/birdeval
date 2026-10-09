"""Compare birdeval's BirdSet clip results with published per-site figures.

    python -I scripts/compare_published.py [RUN_DIR ...] [--ref configs/reference/published.yaml]
                                           [--tol 0.02] [--out results/published_comparison.csv]

RUN_DIR is a results/<run>/ dir (default: every one with results). Rows come
from each pair's summary.json, so a stopped run works too. Only clips-mode rows
(birdset_<SITE>, not _scape) in the `all` scope are used: that is BirdSet's
protocol, which scores every class of the site.

For models with published targets the result is flagged when it falls outside
the targets' range by more than --tol. Context rows (other models on the same
site) are shown for scale only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

METRICS = ["AUROC", "cmAP", "top1"]
CLIPS = re.compile(r"^birdset_([A-Z]{3})$")


def published(ref: dict) -> pd.DataFrame:
    rows = []
    for src, s in ref["sources"].items():
        for model, metrics in s["models"].items():
            for metric, sites in metrics.items():
                for site, v in sites.items():
                    rows.append({"ref": f"{src}/{model}", "metric": metric, "site": site, "value": v})
    return pd.DataFrame(rows)


def load_summaries(run_dirs: list[Path]) -> pd.DataFrame:
    """summary.csv's rows, rebuilt from each pair's summary.json (+ recording_level.json).

    Those are written as each (evalset, model) pair finishes, so a stopped run
    still has them; summary.csv only appears once every eval set is done.
    """
    rows = []
    for d in run_dirs:
        for f in sorted(d.glob("*/*/summary.json")):
            s = json.loads(f.read_text())
            rl = f.with_name("recording_level.json")
            rec = json.loads(rl.read_text()) if rl.exists() else {}
            for scope in ("all", "covered", "shared"):
                m = s[scope]
                lo, hi = m.get("cmAP_ci95") or (float("nan"), float("nan"))
                rows.append({
                    "run": d.name, "evalset": s["evalset"], "model": s["model"], "classes": scope,
                    **{k: m[k] for k in ("cmAP", "AUROC", "top1", "n_segments", "n_classes_scored")},
                    "cmAP_lo": lo, "cmAP_hi": hi,
                    "rec_cmAP": rec.get(scope, {}).get("cmAP", float("nan")),
                })
    return pd.DataFrame(rows)


def ours(run_dirs: list[Path]) -> pd.DataFrame:
    df = load_summaries(run_dirs)
    if df.empty:
        return df
    df["site"] = df["evalset"].str.extract(CLIPS, expand=False)
    df = df[df["site"].notna() & (df["classes"] == "all")]
    return df[["run", "site", "model", *METRICS]].melt(
        id_vars=["run", "site", "model"], var_name="metric", value_name="ours")


def _range(pub: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    sel = pub[pub["ref"].isin(refs)]
    return sel.groupby(["site", "metric"])["value"].agg(["min", "max", "count"]).reset_index()


def compare(our: pd.DataFrame, ref: dict, tol: float) -> pd.DataFrame:
    pub = published(ref)
    ctx = _range(pub, ref.get("context", [])).rename(columns={"min": "context_lo", "max": "context_hi", "count": "n_context"})
    parts = []
    for model, g in our.groupby("model"):
        tgt = _range(pub, ref.get("targets", {}).get(model, []))
        tgt = tgt.rename(columns={"min": "pub_lo", "max": "pub_hi", "count": "n_pub"})
        parts.append(g.merge(tgt, on=["site", "metric"], how="left"))
    df = pd.concat(parts, ignore_index=True).merge(ctx, on=["site", "metric"], how="left")
    # signed distance outside the published range (0 inside it)
    df["delta"] = (df["ours"] - df["pub_hi"]).clip(lower=0) + (df["ours"] - df["pub_lo"]).clip(upper=0)
    df["flag"] = df["delta"].abs() > tol
    df.loc[df["pub_lo"].isna(), "flag"] = False
    return df.sort_values(["site", "model", "metric", "run"]).reset_index(drop=True)


def _fmt(lo, hi) -> str:
    if pd.isna(lo):
        return "-"
    return f"{lo:.3f}" if lo == hi else f"{lo:.3f}-{hi:.3f}"


def show(df: pd.DataFrame) -> None:
    t = pd.DataFrame({
        "run": df["run"], "site": df["site"], "model": df["model"], "metric": df["metric"],
        "ours": df["ours"].map("{:.3f}".format),
        "published": [_fmt(a, b) for a, b in zip(df["pub_lo"], df["pub_hi"])],
        "delta": df["delta"].map(lambda d: "" if pd.isna(d) else f"{d:+.3f}"),
        "flag": df["flag"].map({True: "OUTSIDE", False: ""}),
        "context": [_fmt(a, b) for a, b in zip(df["context_lo"], df["context_hi"])],
    })
    print(t.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--ref", type=Path, default=Path("configs/reference/published.yaml"))
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=Path("results/published_comparison.csv"))
    args = ap.parse_args()

    runs = args.runs or sorted({p.parents[2] for p in Path("results").glob("*/*/*/summary.json")})
    our = ours(runs)
    if our.empty:
        raise SystemExit("no clips-mode BirdSet rows (birdset_<SITE>, scope all) in " + ", ".join(map(str, runs)))
    df = compare(our, yaml.safe_load(args.ref.read_text()), args.tol)
    show(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n = int(df["flag"].sum())
    print(f"\n{n} value(s) outside the published range by more than {args.tol}. Wrote {args.out}")


if __name__ == "__main__":
    main()
