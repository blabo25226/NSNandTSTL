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

## 2026-07-13 05:40

- **作業内容**: Claude Code クラウド環境が **GPU なし・HuggingFace ブロック**と判明（実機確認）。ユーザー指示で二分割対応。**(1) GPU なし再現**: MLP 層寄与を 4 シード再実行＋`aggregate_profiles.py` で集約（全シード k=0 ピーク=浅い MLP では中間層集中は出ない）。スクラッチ小型 Transformer（`tiny_transformer.py`＋`run_tiny_transformer_scan.py`, 6層/合成 modular running-sum）を CPU 実行し層スキャン。**(2) GPU 直前まで**: `run_layer_scan.py` をスタブから全パイプライン CLI に置換、`llm_strategies.py`(Only Bk/Mid-k)・`llm_eval.eval_model` 追加、HF 不要の `Qwen2Config` 凍結テスト追加、notebook/colab_setup を作業ブランチへ更新。
- **変更ファイル**: `TSTL/src/tiny_transformer.py`, `TSTL/src/llm_strategies.py`, `TSTL/src/llm_eval.py`, `TSTL/scripts/{run_tiny_transformer_scan,aggregate_profiles,run_layer_scan}.py`, `TSTL/tests/test_{tiny_transformer,llm_strategies,llm_freeze_hf}.py`, `TSTL/notebooks/{tstl_r1_colab.ipynb,colab_setup.py}`, `TSTL/texts/論文再現計画書.md`, `TSTL/results/*`
- **結果**: pytest **41 件 PASS**（CPU）。小型 Transformer 層寄与（seed42/0 平均, 6層）: C(k)=[0.62, 0.89, 0.95, 0.91, **1.00**, 0.98]。**入力層(k=0)が最弱**、中〜後段の単一層が全層学習をほぼ回復（C≈1.0）=TSTL の定性的特徴（入出力端は低寄与・単層で大部分回復）を CPU で再現。GRPO 本体は GPU+HF マシンで `python scripts/run_layer_scan.py --preset quick` を実行予定（`--dry-run` は CPU 動作確認済み）。
- **メモ**: MLP（浅い回帰）では中間層集中は出ず入力隣接層が支配的。論文の中間層集中は深い事前学習 Transformer 由来と解釈。次段は GPU 実機での R1 実行、または NSN Phase 1（trunk への層寄与）。

## 2026-07-13 06:55

- **作業内容**: GPU 不要でできる論文実証項目の玩具版を CPU 実行（tiny-TF, 6層, 8シード）。**E4 中間層集中**の追試と **E8 ∥Δθ_k∥ vs C(k)** 非相関を実施。共通化のため学習ループを `tiny_transformer.train_model` に集約、相関ヘルパ `utils.pearson_corr/spearman_corr` を追加、`scripts/weight_change_vs_contribution.py`（E8）を新規作成。
- **変更ファイル**: `TSTL/src/{tiny_transformer,utils}.py`, `TSTL/scripts/{run_tiny_transformer_scan,weight_change_vs_contribution}.py`, `TSTL/tests/test_utils_corr.py`, `TSTL/results/{tstl_tiny_tf_seed*,tstl_e8_seed*,tinytf8_aggregate*,tinytf7_noSeed2_aggregate*,tinytf_e8_aggregate*}`
- **結果**:
  - **E4（中間層集中）**: 8シード平均 C(k) は全層 ≈0.92–0.98 で**ほぼフラット**。単一層が全層学習の 9 割超を回復（「1層でほぼ足りる」は頑健に再現）。ただし argmax は **入力層 k=0 が 5/7**、中間深さ(35–65%)は 2/7 のみ → **論文の中間層集中は再現せず**。seed2 は S_full=0.61 と未収束で外れ値。
  - **E8（∥Δθ∥ vs C(k)）**: 相関はシード間で符号バラバラ（Pearson −0.47〜+0.69）、平均 **Pearson 0.02 / Spearman −0.11**（seed2 除外で −0.08/−0.18）。**∥Δθ∥ は C(k) と無相関**＝論文§5の中核洞察（寄与は重み変化量では説明されない）を玩具スケールで再現。∥Δθ‖ の層間相対ばらつきは平均 0.46。
- **メモ**: いずれも玩具（教師あり・非LLM・非GRPO）。実 LLM+GRPO での E3/E4/E5/E6/E7 は GPU+HF 実行が必須で未達。pytest は引き続き緑（相関ヘルパ含む）。

## 2026-07-13 07:30

- **作業内容**: GPU(Windows) 実行手順書を作成し、R1 CLI の実行結果を自動保存するよう補強。`run_layer_scan.py` に stdout/stderr を `run.log` へ複製する Tee、`--dtype {bfloat16,float16,float32}`、戦略スコアの集約を追加。`llm_profile.save_run_report` を新設し `report.json`/`report.md`（S_base/S_full・C(k) 表・Full vs Only Bk/Mid-k 比較）を 1 か所に出力。
- **変更ファイル**: `TSTL/texts/GPU実行手順書.md`(新規), `TSTL/scripts/run_layer_scan.py`, `TSTL/src/llm_profile.py`, `TSTL/tests/test_run_report.py`(新規), `TSTL/daily_report.md`
- **結果**: pytest 48 件 PASS。`--dry-run` は CPU で正常（dtype も反映）。手順書は Windows 前提（venv・CUDA torch・実行/再開・結果の場所・トラブルシュート・R1 判定 E2/E3/E5）。
- **メモ**: 自動保存はローカル + run.log まで（自動 git push はしない方針）。実 GPU 実行はユーザー PC 側。
