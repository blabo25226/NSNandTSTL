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

## 2026-07-13 02:05

- **作業内容**: 論文突き合わせ評価で判明した課題を優先順に改修。
  - **(1) スナップ後 MSE の実測**: `snap.evaluate_snapped()`（葉を one-hot 固定→評価→ソフト重み復元）を追加。pipeline が `snapped_train_mse` を返し `TrainResult` に伝播。
  - **(2) symbolic_ok の再定義**: 旧定義（ソフトモデルのテンプレート照合）を廃し、**export した閉形式=スナップ済みモデルの holdout MSE ≤ 閾値**を成功条件に。`sr_eval` に `snapped_holdout_mse`・`snap_degrade` 列を追加。`match_elementary_template` はスナップ済み予測を受け取る形に変更。
  - **(3) f_prev 再帰（式9）**: `EMLTreeHead(f_prev_mode=...)` を追加。`zero`（既定・v1）と `parent`（1 ステップ top-down 帰還）。`DNNEML.build` に伝播。テスト 2 件追加。
  - **(4) head-capacity 実験**: pipeline に `freeze_trunk_after_search` を追加。`scripts/head_capacity_eval.py`（full/small/freeze/small+parent 比較、soft vs snapped MSE 出力）を新規作成。短ステップ実行で snap 劣化 x2.9〜x13000 を確認（＝現状 trunk 主導を定量化）。
  - **(5) ドキュメント**: `texts/SR検証計画.md` に symbolic 再定義・設計判断（Eq9 softmax 解釈, f_prev, head-capacity, Feynman 合成代替）を追記。
- **変更ファイル**: `NSN/src/snap.py`, `pipeline.py`, `trainer.py`, `simplify.py`, `eml_tree.py`, `model.py`, `scripts/sr_eval.py`, `scripts/head_capacity_eval.py`, `tests/test_eml_tree.py`, `texts/SR検証計画.md`, `NSN/daily_report.md`
- **メモ**: pytest 37 件 PASS。Feynman DB 実データ×深さ{2,3,4}×EQL/KAN 比較は計算時間が長いため別途まとめて実施予定。次: `sr_eval.py` を再実行し新 symbolic 定義での成功率を測る。

## 2026-07-13 02:30

- **作業内容**: スナップ忠実性を実測→改善。`head_capacity_eval` を本ステップ(4000)で実行し、素の soft 学習では snap 劣化が **x17〜x180000** と判明（head でなく trunk が当てはめている）。2 機構で是正:
  - **snap-aware polish**（`snap_aware_polish=True`, 既定）: POLISH 前に離散選択を確定し連続パラメータをその離散木に合わせ込む。→ 全ターゲット faithfulness **x1.0**（snapped=soft）。
  - **gamma-branch マスク**（`EMLTreeHead.effective_logits()`）: zero モードで f_prev=0 の gamma ブランチを argmax から除外。eml の y 引数が `ln(1e-12)≈-27.6` の偽定数を注入する数値発散を解消（`sum`: 12.9→0.025）。
  - 結果（Phase 3, seed=42, 新定義）: **symbolic 5/6**（square/product/exp/sin/sin_plus 成功、sum のみ 0.025 で僅差不合格）。論文中核主張「スナップ後 head が忠実な閉形式」を head レベルで再現。
- **変更ファイル**: `NSN/src/pipeline.py`, `eml_tree.py`, `snap.py`, `tests/test_eml_tree.py`, `texts/SR検証計画.md`, `NSN/daily_report.md`
- **メモ**: pytest 39 件 PASS（gamma-mask テスト 2 件追加）。sum は EML 木で線形和を厳密表現しづらい表現力の限界（数値発散ではない）。次: 残る sum の改善（depth=3 or f_prev=parent の export 対応）か、Feynman DB 本番へ。

## 2026-07-13 03:20

- **作業内容**: 別 AI レビューの指摘②③④に対応（①実 Feynman DB×EQL/KAN は別途）。
  - **② f_prev マスター公式の完全実装＋export 対応**: `parent` モードを **K 回反復（Jacobi）** に一般化
    （`EMLTreeHead.f_prev_passes`, 既定 K=1 で後方互換）。葉 i の f_prev＝親 EML ノード出力という論文定義を
    不動点として K 反復で近づける。`snap.export_symbolic_expression` を parent モードで分岐させ、γ ブランチを
    親ノード式 `eml(兄弟対^(t-1))` に再帰置換（pass0 は const）。**one-hot 重みで export 文字列＝snapped forward が
    厳密一致**することをテストで確認（K=1,2）。旧「1 ステップ近似・export 非対応」を解消。
  - **③ trunk 解釈性（二層解釈の下半分）**: 新 `src/trunk_interpret.py`。`linear_readout`（線形 trunk は厳密
    (W,b)・R²=1、非線形は最小二乗蒸留＋成分別 R²）、`compose_symbolic`（z(x) を head export へ合成し
    **ŷ を x の単一閉形式**に）、`trunk_attribution`（線形 |W|／非線形 mean|∂z/∂x|）。`trunk.py` に
    `linear_weights()` と `num_layers=1` 線形 trunk サポート。`scripts/trunk_interpret_eval.py` 追加。
    線形 trunk では合成閉形式＝snapped model 出力が一致（R²=1）することを実機確認。
  - **④ D≥5 破綻の検証準備**: `EMLTreeHead` の depth 上限を [1,4]→[1,8] に緩和（>4 は `warnings.warn`）。
    `scripts/depth_sweep_eval.py` 追加（depth×seed の成功率・有限率・snapped MSE 中央値を集計）。
- **変更ファイル**: `NSN/src/eml_tree.py`, `snap.py`, `model.py`, `trunk.py`, `src/trunk_interpret.py`(新),
  `scripts/trunk_interpret_eval.py`(新), `scripts/depth_sweep_eval.py`(新),
  `tests/test_eml_tree.py`, `tests/test_trunk_interpret.py`(新), `.gitignore`, `NSN/texts/SR検証計画.md`
- **メモ**: pytest **44 件 PASS**（parent-export 忠実性・不動点残差・depth5・線形合成の 5 件追加）。
  depth スイープ結果（sin/product × seed{0,1} × 1000 step, `results/depth_sweep_20260712_192811/`）:
  **有限率（NaN なし）が D≥5 で急落** — D2/3/4=**1.00**, D5=**0.50**, D6=**0.25**（NaN loss で学習破綻）。
  論文の「D≥5 で学習成功率が急落」を**数値破綻（NaN）として再現**。snapped MSE も D4/D6 で 1e17 級に発散。
  成功率（snapped≤5e-2）は短ステップ設定のため全深さで低く、崩壊の識別子は有限率。

## 2026-07-13 05:40

- **作業内容**: 論文カバレッジ完成の追加フェーズ（残ギャップ A–D）を実装・実験。
  - **B. FLOPs/node コスト解析**: `src/cost.py`。`eml_node_flops()` が超越関数の重み付き合算で
    **論文の ≈111 FLOPs/node を再現**（transcendental=111, +arithmetic=123 total）。`head_flops(depth)`・
    `mlp_flops()`・`scripts/flops_analysis.py`。ハードウェア効率主張の software 再現可能部分を定量化
    （FPGA/アナログ実機合成は範囲外）。
  - **C. `sum` 改善**: `sum` を **zero モード depth 3** に変更 → snapped MSE **3.09e-4**（閾値 1e-2 合格。
    旧 0.025 不合格を解消）。**重要発見**: f_prev の parent モード（K≥2）は**学習モードとしては数値的に不安定**
    （square/product を破壊、sum は NaN）。sum を救うのは f_prev ではなく**木の深さ**。parent の価値は②の
    忠実 export に限定と整理。`trainer.TrainConfig` に `f_prev_mode`/`f_prev_passes`、`sr_eval` に CLI と
    NaN 耐性（1 式失敗で phase 全体を落とさない）を追加。
  - **A+D. 実 Feynman ベンチ×baseline×多シード**: `src/feynman.py`（実 Feynman 12 式, AI Feynman レンジ）、
    `src/baselines.py`（MLP・最小 EQL・任意 KAN）、`scripts/feynman_benchmark.py`（NSN 深さ{2,3,4} vs baseline、
    R²/MSE/複雑度/時間/成功率、多シード集計）。**NSN は feature_dim=4 が安定**（d=6 は exp/ln 発散）。
    代表 5 式×深さ{2,3,4}×seed{0,1} 結果（`results/feynman_bench_20260713_053420/`）:
    **MLP=EQL は成功率 1.00（R²≈0.999）、NSN は d2=0.40 / d3=0.00 / d4=0.10** と脆く、深いほど発散。
    → 論文の優位性主張は**そのままでは再現できず**、式別チューニング or 専用ハード前提を示唆（④ D≥5 崩壊と整合）。
    正直な負の実証結果として記録。
- **変更ファイル**: `NSN/src/{cost,feynman,baselines,trainer}.py`,
  `NSN/scripts/{flops_analysis,feynman_benchmark,sr_eval}.py`,
  `NSN/tests/{test_cost,test_feynman}.py`, `NSN/texts/SR検証計画.md`, `NSN/daily_report.md`
- **メモ**: pytest **50 件 PASS**。phase 3（zero, seed42, `results/sr_phase3_20260713_053541/`）は
  `sum` 合格で **symbolic 5/6**。残る `sin_plus` は**シード敏感**（seed 1/7 で合格 0.01–0.03、seed 0/42/123 で
  不合格 → 成功率 ~40%）。これは本環境（torch 2.13/py3.11）での NSN head の**数値的脆さ**を示し、
  Feynman ベンチの負の結果と整合。以前の環境で sin_plus が合格していたのも同じ脆さの裏返し。
