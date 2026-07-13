"""Verify canonical EML identities from Odrzywołek [1]."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eml import eml  # noqa: E402
from utils import as_complex  # noqa: E402


def main() -> None:
    print("=== EML identity checks ===")

    x = torch.linspace(-1.5, 1.5, 5)
    z = torch.linspace(0.5, 2.5, 5)
    xc, zc = as_complex(x), as_complex(z)
    one = torch.ones_like(xc)

    exp_via_eml = eml(xc, one).real
    exp_direct = torch.exp(x)
    err_exp = (exp_via_eml - exp_direct).abs().max().item()
    print(f"exp: max |eml(x,1) - exp(x)| = {err_exp:.2e}")

    inner_a = eml(one, zc)
    inner_b = eml(inner_a, one)
    ln_via_eml = eml(one, inner_b).real
    ln_direct = torch.log(z)
    err_ln = (ln_via_eml - ln_direct).abs().max().item()
    print(f"ln:  max |eml chain - ln(z)| = {err_ln:.2e}")

    e_const = eml(torch.ones(1), torch.ones(1)).real.item()
    print(f"e = eml(1,1) = {e_const:.6f} (math.e = {math.e:.6f})")

    ok = err_exp < 1e-4 and err_ln < 1e-3
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
