## 2026-07-13 02:45

- **作業内容**: TSTL 作業計画書 `TSTL/texts/作業計画書.md` を初版作成。論文 arXiv:2607.01232（層寄与 C(k)、Only Bk / Boost Bk / Mid-k）を整理し、Phase 0-B は MLP 教師あり回帰での縮小再現と定義。コーディングは承諾待ち。
- **変更ファイル**: `TSTL/texts/作業計画書.md`, `TSTL/daily_report.md`
- **メモ**: ブランチ `20260713_create_TSTL`。論文フルの Qwen3+GRPO は Phase 0-B+ 以降。

## 2026-07-13 02:57

- **作業内容**: Phase 0-B 実装完了。`freeze.py`, `layer_contribution.py`, `strategies.py`, `mlp_bench.py`, `scripts/profile_layers.py`, テスト 11 件。`profile_layers` 実行確認（seed=42: Only B3 R²=0.447 > Full 0.407）。
- **変更ファイル**: `TSTL/src/*`, `TSTL/scripts/profile_layers.py`, `TSTL/tests/*`, `TSTL/requirements.txt`, `TSTL/texts/作業計画書.md`, `TSTL/daily_report.md`
- **メモ**: ヒートマップは matplotlib ではなく PIL 描画（Windows 安定性）。中間層集中は MLP 縮小再現では部分的（Only Bk は論文通り Full 超え）。

## 2026-07-13 03:00

- **作業内容**: 論文再現計画書 `TSTL/texts/論文再現計画書.md` 初版作成（R1–R4 段階、計算資源・完了判定・承諾欄）。
- **変更ファイル**: `TSTL/texts/論文再現計画書.md`, `TSTL/texts/作業計画書.md`, `TSTL/daily_report.md`
- **メモ**: コーディングは承諾後。実用完了は M-R2、完全再現は M-R3/R4。

## 2026-07-13 03:10

- **作業内容**: 論文再現承諾（Colab GPU）。R1 コア: `llm_freeze.py`, `llm_eval.py`, `llm_grpo.py`, `llm_profile.py`, `requirements-r.txt`, `notebooks/tstl_r1_colab.ipynb`, テスト追加。
- **変更ファイル**: `TSTL/src/llm_*.py`, `TSTL/tests/test_llm_*.py`, `TSTL/notebooks/tstl_r1_colab.ipynb`, `TSTL/texts/論文再現計画書.md`, `TSTL/daily_report.md`
- **メモ**: ローカルは Intel Arc + PyTorch CPU。GRPO 実行は Colab のみ。

## 2026-07-13 03:40

- **作業内容**: Colab ノートブック修正。Cursor 拡張は nb のみ送信のため、セル1で git clone + `sys.path` + `pip install -r` を自動化。
- **変更ファイル**: `TSTL/notebooks/tstl_r1_colab.ipynb`, `TSTL/daily_report.md`
- **メモ**: エラー原因は `/content` に `requirements-r.txt` と `src/` が無かったこと。
