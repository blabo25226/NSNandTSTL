"""Resume-friendly layer scan entry point (run on Colab with GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_grpo import GrpoRunConfig, grpo_dependencies_available  # noqa: E402
from llm_profile import default_r1_out_dir, resume_layer_scan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="TSTL R1 layer scan (Colab GPU)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-subset", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    out_dir = args.out_dir or default_r1_out_dir()
    config = {
        "model": args.model,
        "train_subset": args.train_subset,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
    }

    if args.dry_run or not grpo_dependencies_available():
        print(json.dumps({"out_dir": str(out_dir), **config}, indent=2))
        if not grpo_dependencies_available():
            print(
                "\nNote: transformers/trl not installed. "
                "Run this script on Colab with requirements-r.txt."
            )
        return

    # Full Colab implementation hooks in notebooks/tstl_r1_colab.ipynb
    # This CLI resumes from checkpoints written by the notebook.
    raise SystemExit(
        "GPU training is started from notebooks/tstl_r1_colab.ipynb. "
        f"Use --out-dir {out_dir} to resume an existing scan."
    )


if __name__ == "__main__":
    main()
