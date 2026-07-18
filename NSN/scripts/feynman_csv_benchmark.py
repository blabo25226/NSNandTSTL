"""
NSN full Feynman CSV benchmark (samplecode-compatible protocol + Gaussian noise).

Usage:
  python scripts/feynman_csv_benchmark.py --limit 10
  python scripts/feynman_csv_benchmark.py --all
  python scripts/feynman_csv_benchmark.py --eq-ids I.12.1 I.14.3 I.25.13
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feynman_csv import (  # noqa: E402
    FeynmanCsvEq,
    RMSE_PARTIAL_FIXED,
    RMSE_THRESHOLD_FIXED,
    classify_rmse,
    default_csv_path,
    generate_feynman_samples,
    load_feynman_csv,
    noise_aware_rmse_ok_threshold,
    rmse,
    to_sr_target,
)
from leaf_softmax import LeafSoftmaxMode  # noqa: E402
from model import DNNEML  # noqa: E402
from pipeline import OdrzywolekPipelineConfig, run_odrzywolek_pipeline  # noqa: E402
from targets import DEFAULT_NOISE_STD_REL  # noqa: E402
from trainer import TrainConfig, _config_for_target  # noqa: E402
from utils import set_seed  # noqa: E402

BASELINE_EML_SR_OK_RATE = 0.091
BASELINE_PYSR_OK_RATE_LO = 0.50
BASELINE_PYSR_OK_RATE_HI = 0.60


def run_nsn(
    eq: FeynmanCsvEq,
    x: torch.Tensor,
    y_noisy: torch.Tensor,
    y_clean: torch.Tensor,
    train_seed: int,
    noise_std_rel: float,
    steps_override: int | None = None,
) -> dict:
    target = to_sr_target(eq, data_seed=train_seed)
    base = TrainConfig(seed=train_seed, n_train=x.shape[0], noise_std_rel=0.0)
    cfg = _config_for_target(target, base)
    if steps_override is not None:
        cfg.steps = steps_override

    result = {
        "eq_id": eq.filename,
        "status": "failed",
        "status_fixed": "failed",
        "status_noise_aware": "failed",
        "found_expression": None,
        "simplified": None,
        "rmse_clean": None,
        "rmse_noisy": None,
        "rmse_ok_threshold_fixed": RMSE_THRESHOLD_FIXED,
        "rmse_ok_threshold_noise_aware": None,
        "elapsed_s": 0.0,
        "finite": False,
        "head_depth": cfg.head_depth,
        "feature_dim": cfg.feature_dim,
        "steps": cfg.steps,
        "error_msg": None,
    }

    ok_noise_thr = noise_aware_rmse_ok_threshold(y_clean, noise_std_rel)
    result["rmse_ok_threshold_noise_aware"] = ok_noise_thr

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_seed(train_seed)
            model = DNNEML.build(
                input_dim=target.input_dim,
                feature_dim=cfg.feature_dim,
                head_depth=cfg.head_depth,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                leaf_softmax_mode=LeafSoftmaxMode.SOFTMAX,
            )
            pipe_cfg = OdrzywolekPipelineConfig(
                total_steps=cfg.steps,
                lr=cfg.lr,
                polish_lr=cfg.polish_lr,
                weight_decay=cfg.weight_decay,
                seed=train_seed,
            )
            pipe = run_odrzywolek_pipeline(model, x, y_noisy, pipe_cfg)

        with torch.no_grad():
            pred = model(x)
        rmse_clean = rmse(y_clean, pred)
        rmse_noisy = rmse(y_noisy, pred)
        finite = bool(torch.isfinite(pred).all())

        result.update(
            found_expression=pipe.expression_eml,
            simplified=pipe.simplified,
            rmse_clean=rmse_clean if finite else None,
            rmse_noisy=rmse_noisy if finite else None,
            finite=finite,
            status_fixed=classify_rmse(rmse_clean, RMSE_THRESHOLD_FIXED, RMSE_PARTIAL_FIXED)
            if finite
            else "failed",
            status_noise_aware=classify_rmse(rmse_clean, ok_noise_thr, RMSE_PARTIAL_FIXED)
            if finite
            else "failed",
        )
        result["status"] = result["status_fixed"]
    except Exception as exc:
        result["status"] = "error"
        result["status_fixed"] = "error"
        result["status_noise_aware"] = "error"
        result["error_msg"] = str(exc)

    result["elapsed_s"] = time.perf_counter() - t0
    return result


def _fmt_rmse(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.3e}"


def _status_icon(s: str) -> str:
    return {
        "ok": "ok",
        "partial": "partial",
        "failed": "failed",
        "skipped": "skipped",
        "error": "error",
    }.get(s, "?")


def write_report(summary: dict, report_path: Path) -> None:
    results = summary.get("results", [])
    total = summary.get("total", 0)
    settings = summary.get("settings", {})
    total_elapsed = summary.get("total_elapsed_s", 0.0)

    fixed_ok = sum(1 for r in results if r.get("status_fixed") == "ok")
    fixed_partial = sum(1 for r in results if r.get("status_fixed") == "partial")
    noise_ok = sum(1 for r in results if r.get("status_noise_aware") == "ok")
    noise_partial = sum(1 for r in results if r.get("status_noise_aware") == "partial")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# NSN Feynman Equations 全式推定レポート",
        "",
        f"**生成日時:** {now_str}",
        "",
        "---",
        "",
        "## 1. 実験概要",
        "",
        "### 実験設定",
        "",
        "| パラメータ | 値 |",
        "|-----------|-----|",
        f"| N_SAMPLES | {settings.get('N_SAMPLES')} |",
        f"| noise_std_rel | {settings.get('noise_std_rel')} |",
        f"| RMSE ok (固定) | {settings.get('RMSE_THRESHOLD_FIXED')} |",
        f"| RMSE partial (固定) | {settings.get('RMSE_PARTIAL_FIXED')} |",
        f"| train_seed | {settings.get('train_seed')} |",
        f"| 合計実行時間 | {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min) |",
        "",
        "---",
        "",
        "## 2. 総合結果サマリー",
        "",
        "| 指標 | 固定閾値 | ノイズ考慮閾値 |",
        "|-----|---------|---------------|",
        f"| 対象方程式数 | {total} | {total} |",
        f"| ok | {fixed_ok} ({fixed_ok / total * 100:.1f}%)" if total else "| ok | 0 | 0 |",
    ]
    if total:
        lines[-1] += f" | {noise_ok} ({noise_ok / total * 100:.1f}%) |"
    else:
        lines[-1] = "| ok | 0 | 0 |"

    lines += [
        f"| partial | {fixed_partial} ({fixed_partial / total * 100:.1f}%)" + (
            f" | {noise_partial} ({noise_partial / total * 100:.1f}%) |" if total else " | 0 |"
        ),
        f"| skipped | {skipped} | {skipped} |",
        "",
        "### ベースライン比較（固定 ok 閾値）",
        "",
        "| 手法 | ok 率 | 備考 |",
        "|------|-------|------|",
        f"| eml-sr | {BASELINE_EML_SR_OK_RATE * 100:.1f}% | samplecode レポート（ノイズなし） |",
        f"| PySR | {BASELINE_PYSR_OK_RATE_LO * 100:.0f}–{BASELINE_PYSR_OK_RATE_HI * 100:.0f}% | ユーザー報告 |",
        f"| **NSN (本実験)** | **{fixed_ok / total * 100:.1f}%** | noise_std_rel={settings.get('noise_std_rel')} |"
        if total
        else "| **NSN (本実験)** | N/A | |",
        "",
        "---",
        "",
        "## 3. 各方程式の詳細結果",
        "",
        "| # | 式ID | 変数数 | status_fixed | status_noise | RMSE(clean) | NSN 出力 | 時間(s) |",
        "|---|------|--------|--------------|--------------|-------------|----------|---------|",
    ]

    for r in results:
        expr = str(r.get("found_expression") or "N/A")
        if len(expr) > 50:
            expr = expr[:47] + "..."
        lines.append(
            f"| {r.get('index', '?')} | `{r.get('eq_id', '?')}` | {r.get('n_vars', '?')} | "
            f"{_status_icon(r.get('status_fixed', '?'))} | {_status_icon(r.get('status_noise_aware', '?'))} | "
            f"{_fmt_rmse(r.get('rmse_clean'))} | `{expr}` | {r.get('elapsed_s', 0):.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. 変数数別成功率（固定閾値）",
        "",
        "| 変数数 | 対象数 | ok | partial | 合計回収率 |",
        "|--------|--------|-----|---------|-----------|",
    ]

    var_stats: dict[int, dict[str, int]] = {}
    for r in results:
        if r.get("status") == "skipped":
            continue
        nv = int(r.get("n_vars", 0))
        st = r.get("status_fixed", "failed")
        if nv not in var_stats:
            var_stats[nv] = {"ok": 0, "partial": 0, "total": 0}
        var_stats[nv]["total"] += 1
        if st == "ok":
            var_stats[nv]["ok"] += 1
        elif st == "partial":
            var_stats[nv]["partial"] += 1

    for nv in sorted(var_stats):
        t = var_stats[nv]["total"]
        o = var_stats[nv]["ok"]
        p = var_stats[nv]["partial"]
        rate = (o + p) / t * 100 if t else 0
        lines.append(f"| {nv} | {t} | {o} | {p} | {rate:.0f}% |")

    lines += [
        "",
        "---",
        "",
        "## 5. 考察",
        "",
        "- 固定 RMSE < 1e-4 は 1% 相対ノイズ下ではほぼ達成不可能。ノイズ考慮列を併記。",
        "- eml-sr / PySR ベースラインはノイズなし（または別設定）の可能性があり、直接比較は参考値。",
        "",
        f"*本レポートは `feynman_csv_benchmark.py` により自動生成されました。*",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="NSN Feynman CSV benchmark")
    parser.add_argument("--csv", type=str, default=None, help="Path to FeynmanEquations.csv")
    parser.add_argument("--all", action="store_true", help="Run all equations in CSV")
    parser.add_argument("--limit", type=int, default=None, help="Run first N equations")
    parser.add_argument("--eq-ids", nargs="+", default=None, help="Specific equation IDs")
    parser.add_argument("--n-samples", type=int, default=750)
    parser.add_argument("--noise-std-rel", type=float, default=DEFAULT_NOISE_STD_REL)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--data-seed-base", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else default_csv_path()
    equations = load_feynman_csv(csv_path)
    if args.eq_ids:
        id_set = set(args.eq_ids)
        equations = [e for e in equations if e.filename in id_set]
    elif args.limit is not None:
        equations = equations[: args.limit]
    elif not args.all:
        parser.error("Specify --all, --limit N, or --eq-ids ...")

    print("=" * 62)
    print("  NSN: Feynman CSV Benchmark")
    print(f"  N_SAMPLES={args.n_samples}  noise_std_rel={args.noise_std_rel}")
    print("=" * 62)
    print(f"[INFO] Loaded {len(equations)} equations from {csv_path}")

    results: list[dict] = []
    total_start = time.time()

    for idx, eq in enumerate(equations):
        eq_id = eq.filename
        print(f"\n[{idx + 1:03d}/{len(equations)}] {eq_id}  (vars={eq.n_vars})")

        data_seed = args.data_seed_base + idx
        samples = generate_feynman_samples(
            eq, args.n_samples, seed=data_seed, noise_std_rel=args.noise_std_rel
        )
        if samples is None:
            print(f"  [SKIP] too few valid samples")
            results.append(
                {
                    "index": idx + 1,
                    "eq_id": eq_id,
                    "n_vars": eq.n_vars,
                    "var_names": eq.var_names,
                    "status": "skipped",
                    "status_fixed": "skipped",
                    "status_noise_aware": "skipped",
                    "reason": "too_few_valid_samples",
                    "found_expression": None,
                    "rmse_clean": None,
                    "elapsed_s": 0.0,
                }
            )
            continue

        x, y_noisy, y_clean = samples
        result = run_nsn(
            eq,
            x,
            y_noisy,
            y_clean,
            train_seed=args.train_seed,
            noise_std_rel=args.noise_std_rel,
            steps_override=args.steps,
        )
        result["index"] = idx + 1
        result["n_vars"] = eq.n_vars
        result["var_names"] = eq.var_names
        results.append(result)

        rmse_str = _fmt_rmse(result.get("rmse_clean"))
        print(
            f"  RMSE(clean): {rmse_str}  fixed={result['status_fixed']}  "
            f"noise_aware={result['status_noise_aware']}  Time: {result['elapsed_s']:.1f}s"
        )
        if result.get("found_expression"):
            expr = result["found_expression"]
            if len(expr) > 60:
                expr = expr[:57] + "..."
            print(f"  Expr: {expr}")

    total_elapsed = time.time() - total_start
    fixed_ok = sum(1 for r in results if r.get("status_fixed") == "ok")

    summary = {
        "total": len(equations),
        "ok_fixed": fixed_ok,
        "ok_noise_aware": sum(1 for r in results if r.get("status_noise_aware") == "ok"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "total_elapsed_s": total_elapsed,
        "settings": {
            "N_SAMPLES": args.n_samples,
            "noise_std_rel": args.noise_std_rel,
            "RMSE_THRESHOLD_FIXED": RMSE_THRESHOLD_FIXED,
            "RMSE_PARTIAL_FIXED": RMSE_PARTIAL_FIXED,
            "train_seed": args.train_seed,
            "data_seed_base": args.data_seed_base,
            "csv": str(csv_path),
        },
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"feynman_csv_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, out_dir / "summary.md")

    print("\n" + "=" * 62)
    print(f"  SUMMARY: {fixed_ok} / {len(equations)} equations ok (fixed threshold)")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    print(f"  Results -> {out_dir}")
    print("=" * 62)


if __name__ == "__main__":
    main()
