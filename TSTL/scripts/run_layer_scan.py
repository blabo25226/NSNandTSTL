"""TSTL R1 end-to-end layer scan (run on a GPU + HuggingFace machine).

Consolidates the notebook logic into one CLI so a GPU PC can reproduce Phase R1
with a single command. Everything up to the actual GPU/HF training is validated
on CPU via ``--dry-run`` (no model/dataset download).

    # GPU + internet (HuggingFace) machine:
    python scripts/run_layer_scan.py --preset quick
    python scripts/run_layer_scan.py --model Qwen/Qwen2.5-0.5B-Instruct --preset standard

    # Any machine (no GPU/HF needed): print the plan and cost estimate
    python scripts/run_layer_scan.py --preset quick --dry-run

Pipeline: S_base -> Full GRPO -> per-layer scan (resume-friendly) -> C(k) +
depth plot -> Only Bk / Mid-k strategy comparison -> results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_grpo import GrpoRunConfig, grpo_dependencies_available  # noqa: E402
from llm_profile import default_r1_out_dir, resume_layer_scan  # noqa: E402
from r1_presets import (  # noqa: E402
    FULL,
    QUICK,
    STANDARD,
    R1Preset,
    estimate_grpo_forward_passes,
    layer_indices_to_scan,
)

PRESETS: dict[str, R1Preset] = {"quick": QUICK, "standard": STANDARD, "full": FULL}


class _Tee:
    """Duplicate a stream to a log file so results survive disconnects."""

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()


def _resolve_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_config(preset: R1Preset, args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "preset": preset.name,
        "train_subset": preset.train_n,
        "eval_subset": preset.eval_n,
        "steps": preset.steps,
        "num_generations": preset.num_generations,
        "max_completion_length": preset.max_completion_length,
        "layer_stride": preset.layer_stride,
        "lr": args.lr,
        "seed": args.seed,
        "dtype": args.dtype,
    }


def run_pipeline(args: argparse.Namespace, preset: R1Preset, out_dir: Path) -> None:
    """Actual GPU/HF run. Imports heavy deps lazily so --dry-run stays light."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_data import load_gsm8k_subset, to_grpo_rows
    from llm_eval import eval_model
    from llm_freeze import num_transformer_layers
    from llm_profile import save_run_report
    from llm_strategies import strategy_layer_indices

    out_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = out_dir / "base_model"
    dtype = _resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_split, eval_split = load_gsm8k_subset(preset.train_n, preset.eval_n)
    grpo_rows = to_grpo_rows(train_split)

    def load_model(src: str) -> "torch.nn.Module":
        return AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=dtype, device_map="auto"
        )

    def make_config(layer_indices, output_dir, save_model):
        return GrpoRunConfig(
            output_dir=output_dir,
            learning_rate=args.lr,
            num_train_steps=preset.steps,
            num_generations=preset.num_generations,
            max_completion_length=preset.max_completion_length,
            seed=args.seed,
            train_layer_indices=layer_indices,
            tokenizer=tokenizer,
            save_model=save_model,
        )

    # S_base: base model saved once so every single-layer run starts identically.
    base_model = load_model(args.model)
    base_model.save_pretrained(base_ckpt)
    tokenizer.save_pretrained(base_ckpt)
    s_base = eval_model(base_model, eval_split, tokenizer, max_new_tokens=preset.max_completion_length)
    print(f"S_base = {s_base:.4f}")

    # S_full: full-parameter GRPO.
    from llm_grpo import run_grpo_train

    full_model = load_model(str(base_ckpt))
    run_grpo_train(full_model, grpo_rows, make_config(None, out_dir / "full", True))
    s_full = eval_model(full_model, eval_split, tokenizer, max_new_tokens=preset.max_completion_length)
    print(f"S_full = {s_full:.4f}")

    n_layers = num_transformer_layers(full_model)
    scan_layers = layer_indices_to_scan(n_layers, preset.layer_stride)
    print(f"layers={n_layers} scan={scan_layers}")

    def train_and_eval_layer(k: int) -> float:
        m = load_model(str(base_ckpt))
        run_grpo_train(m, grpo_rows, make_config([k], out_dir / f"layer_{k}", False))
        return eval_model(m, eval_split, tokenizer, max_new_tokens=preset.max_completion_length)

    result = resume_layer_scan(
        out_dir,
        n_layers,
        train_and_eval_layer,
        s_base=s_base,
        s_full=s_full,
        layer_indices=scan_layers,
        config=build_config(preset, args),
    )
    print(f"best C = {max(result.contributions.values()):.3f}")

    # §4 strategies: Only Bk / Mid-k vs Full (uses scanned contributions).
    strategies: dict[str, dict] = {}
    for strat in ("only_bk", "mid_k"):
        idx = strategy_layer_indices(strat, n_layers, args.strategy_k, contributions=result.contributions)
        m = load_model(str(base_ckpt))
        run_grpo_train(m, grpo_rows, make_config(idx, out_dir / f"strategy_{strat}", False))
        s = eval_model(m, eval_split, tokenizer, max_new_tokens=preset.max_completion_length)
        print(f"strategy {strat} (layers={idx}): S = {s:.4f}")
        strategies[strat] = {"layers": idx, "score": s}
        (out_dir / f"strategy_{strat}.json").write_text(
            json.dumps({"strategy": strat, **strategies[strat]}, indent=2),
            encoding="utf-8",
        )

    # One consolidated report (report.json + report.md) on top of the per-artifact files.
    save_run_report(
        out_dir,
        config=build_config(preset, args),
        s_base=s_base,
        s_full=s_full,
        s_per_layer=result.s_per_layer,
        contributions=result.contributions,
        strategies=strategies,
    )
    print(f"results -> {result.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TSTL R1 end-to-end layer scan")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--preset", choices=list(PRESETS), default="quick")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy-k", type=int, default=3, help="k for Only Bk / Mid-k")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model compute dtype (use float16 on GPUs without bf16 support)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit (no GPU/HF)")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    out_dir = args.out_dir or default_r1_out_dir()
    config = build_config(preset, args)

    # A nominal 28-layer model (Qwen2.5-0.5B) for the pre-flight cost estimate.
    nominal_layers = 24
    scan = layer_indices_to_scan(nominal_layers, preset.layer_stride)
    est = estimate_grpo_forward_passes(preset, len(scan))

    if args.dry_run or not grpo_dependencies_available():
        plan = {
            "out_dir": str(out_dir),
            **config,
            "nominal_scan_layers": scan,
            "estimated_grpo_generation_passes": est,
        }
        print(json.dumps(plan, indent=2))
        if not grpo_dependencies_available():
            print(
                "\nNote: transformers/trl not installed (or no GPU). "
                "This is a dry run only. Install TSTL/requirements-r.txt and "
                "run on a GPU machine for the real scan."
            )
        return

    # Auto-save: mirror stdout/stderr to out_dir/run.log so results survive disconnects.
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(orig_out, log_file)
        sys.stderr = _Tee(orig_err, log_file)
        try:
            print(f"# run.log -> {log_path}")
            run_pipeline(args, preset, out_dir)
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err


if __name__ == "__main__":
    main()
