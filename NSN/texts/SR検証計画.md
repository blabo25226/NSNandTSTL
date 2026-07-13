# NSN シンボリック回帰（SR）検証計画

Phase 0-A 完了後の **(1)〜(3)** 検証手順。非単調関数は失敗を許容し、一般的な初等関数で SR できるかを確認する。

## 評価の2段階

| 段階 | 意味 | 成功条件 |
|------|------|----------|
| **数値 (numeric)** | 学習した（ソフトな）DNN-EML が真の式に近似できているか | soft holdout MSE ≤ 閾値 |
| **記号 (symbolic)** | **エクスポートした閉形式 EML 式**（=スナップ済みモデル）が忠実か | **snapped holdout MSE ≤ 閾値** |

### symbolic 指標の再定義（2026-07-13）

**旧定義（〜2026-07-13 早朝）は誤りだった。** 旧 `symbolic_ok = numeric_ok AND (template_guess == true_formula)` は、
**ソフトなブラックボックスモデル**の出力を 7 個のテンプレート辞書と照合していただけで、
export した EML 記号式を一切検証していなかった。論文の主張「スナップ後の head が忠実な閉形式になる」を測っていない。

**新定義**: `evaluate_snapped()`（`src/snap.py`）で葉ロジットを one-hot に固定した
**スナップ済みモデル**の holdout MSE を測り、閾値内なら symbolic OK とする。
スナップ済みモデルの出力は `export_symbolic_expression()` を z=trunk(x) 上で数値評価した値に一致するため、
これは「export した閉形式が忠実か」を直接測る。`template_guess` は補助情報として残す（スナップ済み出力に対して計算）。

サマリには `snapped_holdout_mse` と `snap_degrade`（= snapped/soft MSE 比）を出力する。
比が 1 に近いほどスナップが成功。現状は比が大きく、**head ではなく trunk が当てはめを担っている**ことが定量化される。

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

## 最新結果（新定義, seed=42, noise_std_rel=0.01, 2026-07-13）

**snap-aware polish + gamma-mask** 導入後。symbolic OK は **スナップ済み閉形式の holdout MSE ≤ 閾値**（忠実性）で判定。
faithfulness は全ターゲットで **x1.0**（snapped MSE = soft MSE）＝スナップは構成上ロスレス。

| target | true | snapped holdout MSE | symbolic OK |
|--------|------|---------------------|-------------|
| square | `x0^2` | 4.8e-4 | ✅ |
| product | `x0*x1` | 1.4e-4 | ✅ |
| sum | `x0+x1` | 2.5e-2 | ❌（閾値1e-2に僅差。EML木で線形和の厳密表現が本質的に困難） |
| exp | `exp(x0)` | 1.2e-4 | ✅ |
| sin | `sin(x0)` | 6.8e-4 | ✅ |
| sin_plus | `sin(x0)+x1` | 3.0e-4 | ✅ |
| **Phase 3 合計** | | | **5/6** |

論文の中核主張「スナップ後の head が忠実な閉形式になる」を head レベルで再現（5/6）。
残る `sum` は数値発散ではなく表現力の僅差（0.025 vs 0.01）。

### スナップ忠実性を実現した2つの機構（2026-07-13）

1. **snap-aware polish**（`OdrzywolekPipelineConfig.snap_aware_polish=True`, 既定）:
   POLISH の前に離散選択を確定（argmax→one-hot）し、連続パラメータ（α/β/trunk）を**その離散木に合わせ込む**。
   これで学習対象＝export される木そのものになり、snapped MSE ≈ soft MSE。
   導入前は snap 劣化が **x17〜x180000**（`head_capacity_eval` 参照）だった。
2. **gamma-branch マスク**（`EMLTreeHead.effective_logits()`）:
   zero モードでは f_prev=0 のため gamma ブランチが無意味かつ有害
   （eml の y 引数になると `ln(1e-12)≈-27.6` の偽定数を注入）。zero モードでは gamma を argmax 対象から除外。
   これで `sum` の数値発散（12.9→0.025）を解消。

## 旧「結果」（seed=42, 2026-07-13 早朝、参考）

> ⚠️ 下表 symbolic OK 9/9 は **旧・誤定義**（ソフトモデルのテンプレート照合）による値。
> スナップ済み式は検証しておらず、実際にはスナップが忠実でなかった（上の新結果で是正）。

| Phase | numeric OK | symbolic OK (旧定義) | 学習時間 | 壁時計 |
|-------|------------|----------------------|----------|--------|
| 1 | 1/1 | 1/1 | 37.0s | 37.0s |
| 2 | 2/2 | 2/2 | 75.0s | 75.0s |
| 3 | 6/6 | 6/6 | 191.8s | 191.9s |
| **合計** | **9/9** | **9/9（旧定義）** | **~304s** | **~304s** |

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

## 設計判断（2026-07-13、論文突き合わせ後の改修）

論文（arXiv:2604.13871）は参照実装を持たない提案論文のため、以下は本実装での解釈・拡張を明記する。

### 1. 葉のアフィン式（論文 Eq. 9）の softmax 解釈

論文 `l_i(z) = α_i + β_i^T z + γ_i·f_prev` は 3 項の**和**。本実装は 3 項を **softmax 選択（凸結合）**
`w_a·α + w_b·(β^T z) + w_g·f_prev` として実装している（`eml_tree.py`）。
これは葉を単体頂点にスナップ（=どの項型を選ぶか決定）可能にするための再解釈であり、論文の字義（自由な和）とは異なる。

### 2. f_prev の扱い（`f_prev_mode`）

論文の f_prev は「親 EML ノードの出力（根では 0）」＝ Odrzywołek のマスター公式の再帰項。
完全二分木のボトムアップ評価では葉の親は未計算のため、この再帰は循環する。本実装は 2 モードを提供:

- **`zero`（デフォルト・v1）**: 全葉で f_prev = 0。ボトムアップ評価と整合。既存結果・スナップ export はこのモード。
- **`parent`（実験用）**: f_prev=0 の 1 パス目で各葉の親ノード出力を計算し、γ·(親出力) を加えて葉を再計算する
  **1 ステップ top-down 帰還**（1 回の Jacobi 反復）。厳密な不動点ではない。記号 export は未対応（`zero` 前提）。

### 3. head-capacity 研究（`scripts/head_capacity_eval.py`）

「head が記号を復元しているのか、trunk MLP が全部当てはめているのか」を切り分ける。
`full` / `small`（小型 trunk）/ `freeze`（SEARCH 後に trunk 凍結）/ `small+parent` を比較し、
各構成の soft MSE と snapped MSE、劣化比を出力する。劣化比が 1 に近いほど head が忠実に記号を担っている。
pipeline の `freeze_trunk_after_search` フラグが対応する。

### 4. Feynman DB は合成代替（暫定）

現状の `feynman_*` ターゲットは 2 変数の合成式（`x0*x1`, `1/x0^2`）で、実 Feynman DB ではない。
論文ロードマップ項目 2（実 Feynman DB × 深さ {2,3,4} × EQL/KAN 比較）は計算時間が長いため別途まとめて実施する。

## 関連ファイル

- `src/targets.py` — ターゲット定義
- `src/trainer.py` — 学習ループ
- `src/simplify.py` — 簡約・テンプレート照合・記号忠実性評価
- `src/snap.py` — スナップ／`evaluate_snapped`（記号式の数値評価）
- `src/eml_tree.py` — EML 木 head（`f_prev_mode`）
- `scripts/sr_eval.py` — 評価実行（snapped MSE 列を含む）
- `scripts/head_capacity_eval.py` — head vs trunk 容量研究
