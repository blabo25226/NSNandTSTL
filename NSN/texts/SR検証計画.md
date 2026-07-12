# NSN シンボリック回帰（SR）検証計画

Phase 0-A 完了後の **(1)〜(3)** 検証手順。非単調関数は失敗を許容し、一般的な初等関数で SR できるかを確認する。

## 評価の2段階

| 段階 | 意味 | 成功条件 |
|------|------|----------|
| **数値 (numeric)** | 学習した DNN-EML が真の式に近似できているか | holdout MSE ≤ 閾値 |
| **記号 (symbolic)** | 数値成功かつ、テンプレート照合で真の式と一致 | `template_guess == true_formula` |

### ラベルノイズ（2026-07-13 追加）

教師信号 `y` にガウシアン誤差を付加する（デフォルト **1%**）:

```
y_noisy = y_true + noise_std_rel * std(y_true) * ε,   ε ~ N(0, 1)
```

- デフォルト: `noise_std_rel = 0.01`（`targets.DEFAULT_NOISE_STD_REL`）
- 無ノイズ: `--noise-std-rel 0`
- 以前の評価は **誤差なし**（`y = f(x)` の厳密値）だった

閾値:
- 単調な式: `1e-2`（`--mse-strict`）
- 非単調な式: `5e-2`（`--mse-relaxed`）

## フェーズ構成

| Phase | 内容 | ターゲット |
|-------|------|------------|
| **1** | sin_plus 改善 | `sin(x0) + x1` |
| **2** | Feynman 風 1〜2 式 | `x0*x1`（I.29 系積）、`1/x0^2`（I.9 系） |
| **3** | 初等関数スイート | `x0^2`, `x0*x1`, `x0+x1`, `exp(x0)`, `sin(x0)`, `sin_plus` |

## 実行方法

```powershell
cd NSN
python scripts/sr_eval.py --phase 1 --seed 42
python scripts/sr_eval.py --phase 2 --seed 42
python scripts/sr_eval.py --phase 3 --seed 42
```

結果: `NSN/results/sr_phase{N}_<timestamp>/summary.json` と `summary.md`

## 式簡約 (3)

- `src/simplify.py`: EML 文字列の一部ルール簡約 + 初等テンプレート照合
- 完全な `eml → sin` 変換は未実装（Odrzywołek 全ルールは今後）

## 最新結果（seed=42, noise_std_rel=0.01, 2026-07-13）

| Phase | numeric OK | symbolic OK | 学習時間 | 壁時計 |
|-------|------------|-------------|----------|--------|
| 1 | 1/1 | 1/1 | 37.0s | 37.0s |
| 2 | 2/2 | 2/2 | 75.0s | 75.0s |
| 3 | 6/6 | 6/6 | 191.8s | 191.9s |
| **合計** | **9/9** | **9/9** | **~304s** | **~304s** |

### 精度改善の要点（2026-07-13）

1. **AdamW**（`weight_decay=1e-4`）でノイズ付きラベルへの過学習を抑制
2. **ターゲット別ハイパーパラメータ**（例: `exp` は `head_depth=1`、`sin` は `feature_dim=6`）
3. **スナップは式エクスポートのみ** — 推論はソフトマックス重みのまま（ハードスナップは holdout 精度を大幅に悪化させる）
4. `sr_eval.py` に学習時間・壁時計を記録

### 以前（noise-free, 2026-07-12）

| Phase | numeric OK | symbolic OK | メモ |
|-------|------------|-------------|------|
| 1 | 0/1 | 0/1 | sin_plus MSE≈0.47 |
| 2 | 1/2 | 1/2 | 積のみ成功 |
| 3 | 1/6 | 1/6 | 積のみ完全成功 |

## 関連ファイル

- `src/targets.py` — ターゲット定義
- `src/trainer.py` — 学習ループ
- `src/simplify.py` — 簡約・テンプレート照合
- `scripts/sr_eval.py` — 評価実行
