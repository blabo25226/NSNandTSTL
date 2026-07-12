## 2026-07-12 23:26

- **作業内容**: ブランチ `20260712_create_NSN` 上で `NSN/texts/作業計画書.md` を作成。論文から DNN-EML 仕様を整理し、PyTorch 実装のステップ・ファイル構成・検証計画を記載。
- **変更ファイル**: `NSN/texts/作業計画書.md`
- **メモ**: NSN 公式コードは非公開。Odrzywołek [1] の eml-rust 等は参照のみ。

## 2026-07-12 23:32

- **作業内容**: ユーザー指示に基づき NSN 作業ログを本ファイル（`NSN/daily_report.md`）に統一。`AGENTS.md`・作業計画書を更新。§7・§10 の設計判断を承諾済みと記録。コーディングは未着手（待機）。
- **変更ファイル**: `NSN/daily_report.md`, `NSN/texts/作業計画書.md`, `AGENTS.md`
- **メモ**: コーディング開始はユーザーからの明示指示後。

## 2026-07-12 23:39

- **作業内容**: ユーザー承諾に基づき NSN Phase 0 実装を開始。`NSN/src/` に eml, eml_tree, trunk, model, snap, utils を実装。tests 11 件・`train_demo.py`（square, MSE≈5e-4）・`verify_eml_identities.py` を追加。
- **変更ファイル**: `NSN/src/*`, `NSN/tests/*`, `NSN/scripts/*`, `NSN/requirements.txt`, `NSN/texts/作業計画書.md`, `NSN/daily_report.md`
- **メモ**: Step 2 の手動 EML 式再現テスト・Step 5 のスナップテストは未完了。

## 2026-07-12 23:44

- **作業内容**: Step 2/5 完了。`eml_depth2` 手動再現テスト、記号式 `export_symbolic_expression`、`test_snap.py`、`train_demo` の `expression.txt` 出力。解析ヤコビアン検証（式15）。pytest 18 件 PASS。sin_plus デモ学習調整。
- **変更ファイル**: `NSN/src/eml_tree.py`, `NSN/src/snap.py`, `NSN/scripts/train_demo.py`, `NSN/tests/*`, `NSN/texts/作業計画書.md`, `NSN/daily_report.md`
- **メモ**: Phase 0-A（NSN 単独最小実装）完了。

## 2026-07-12 23:56

- **作業内容**: SR 検証 (1)〜(3) 用に `targets.py`, `trainer.py`, `simplify.py`, `scripts/sr_eval.py`, `texts/SR検証計画.md` を追加。Phase 1–3 を seed=42 で実行。積 `x0*x1` は数値・記号とも成功。sin_plus は MSE≈0.47 で部分失敗。
- **変更ファイル**: `NSN/src/targets.py`, `trainer.py`, `simplify.py`, `NSN/scripts/sr_eval.py`, `NSN/tests/test_simplify.py`, `NSN/texts/SR検証計画.md`, `NSN/daily_report.md`
- **メモ**: 完全な eml→初等関数簡約は今後の課題。

## 2026-07-13 00:05

- **作業内容**: SR 用合成データにラベルノイズを追加（デフォルト `noise_std_rel=0.01`）。`targets.py`, `trainer.py`, `sr_eval.py` を更新。以前は誤差なしの厳密 `y=f(x)` だったことを文書化。
- **変更ファイル**: `NSN/src/targets.py`, `trainer.py`, `NSN/scripts/sr_eval.py`, `NSN/tests/test_targets.py`, `NSN/texts/SR検証計画.md`, `NSN/daily_report.md`
- **メモ**: `train_demo.py` は引き続きノイズなし（overfit デモ用）。

## 2026-07-13 01:20

- **作業内容**: SR 精度改善。AdamW 正則化・ターゲット別 HP 調整・スナップを式 export のみに分離。誤差付きデータ（1%）で Phase 1–3 全ターゲット数値・記号成功（9/9）。`sr_eval` に計算時間計測を追加。
- **変更ファイル**: `NSN/src/trainer.py`, `snap.py`, `model.py`, `NSN/scripts/sr_eval.py`, `NSN/texts/SR検証計画.md`, `NSN/daily_report.md`
- **メモ**: ハードスナップ後の推論は holdout MSE が ~1000× 悪化。推論は学習済みソフト重みを使用。

## 2026-07-13 01:25

- **作業内容**: `train_demo.py` に Feynman 1式デモ（`--demo feynman`, I.29 系 `x0*x1`）を追加。overfit（train=test、ノイズなし）。
- **変更ファイル**: `NSN/scripts/train_demo.py`, `NSN/daily_report.md`

## 2026-07-13 01:40

- **作業内容**: Gumbel-softmax / softmax 切替（`leaf_softmax.py`, `--leaf-softmax`）。Odrzywołek 理論モジュール（`odrzywolek.py`）と 4 段パイプライン（`pipeline.py`: SEARCH→HARDEN→POLISH→SNAP）を実装。`trainer.py` はパイプライン経由に統一。pytest 35 件 PASS。
- **変更ファイル**: `NSN/src/leaf_softmax.py`, `odrzywolek.py`, `pipeline.py`, `eml_tree.py`, `model.py`, `trainer.py`, `simplify.py`, `scripts/sr_eval.py`, `train_demo.py`, `verify_odrzywolek.py`, `tests/test_*.py`, `NSN/daily_report.md`
