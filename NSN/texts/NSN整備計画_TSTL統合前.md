# NSN 整備計画書（TSTL 統合前ベースライン確立）

| 項目 | 内容 |
|------|------|
| 作成日 | 2026-07-13 |
| 前提ブランチ | `claude/nsn-reproduction-issues-zvmubu`（PR #2 相当の再現完了状態） |
| 作業ディレクトリ | `NSN/`（TSTL 本体は本フェーズ対象外） |
| 参照論文 | Ipek (2026) arXiv:2604.13871 **（2026-04-21 著者取り下げ済み・v1 原稿を再現対象とする）** |
| 次フェーズ | TSTL 層寄与 `C(k)` の NSN trunk への適用（`TSTL/` + ルート `src/hybrid/`） |
| ステータス | **§10 承諾済み（2026-07-13）— Step 4 ゲート実行完了、Gate A 未達・Gate C 達成** |
| Feynman データ | ルート `data/FeynmanEquations.csv`（本フェーズのゲート対象外・TSTL 後の拡張用） |

---

## 0. 本計画の位置づけ

### 0.1 ユーザー合意の整理

> TSTL なしでも NSN は **ある程度まともなシンボリック回帰ができる状態** にあった方がよい。

本計画はその「ある程度まとも」を **定量ゲート** として定義し、達成するための **最小限の整備**（論文の大改造ではない）を行う。

TSTL は NSN の大改造の**前提条件ではない**が、次を満たさないと層寄与 `C(k)` の測定がノイズだらけになり、統合実験の解釈ができない。

### 0.2 TSTL 統合に必要な NSN 側の最低条件

| 条件 | 理由 |
|------|------|
| 安定ターゲットで **再現可能な symbolic OK** | `S_base`, `S_full`, `S_k` の差が意味を持つ |
| **有限率（NaN なし）** が高い | 層スキャンで全滅しない |
| **スナップ忠実性**（degrade ≈ 1）がプロトコル上保証 | head が記号を担っているかの監査 |
| **評価プロトコル固定** | TSTL 前後で比較可能 |

Feynman 12 式での MLP 超えは **本フェーズのスコープ外**。

---

## 1. 現状サマリ（2026-07-13 再現結果）

| 項目 | 状態 |
|------|------|
| 手法本体（EML・DNN-EML・スナップ・FLOPs） | 再現済み |
| Phase 3（seed=42, 式別 HP） | symbolic **5/6**（`sin_plus` 不合格） |
| Phase 3（多シード未集計） | `sin_plus` は seed 依存（成功率 ~40% と記録） |
| Feynman ベンチ | MLP/EQL に大敗（正直な負の結果） |
| head vs trunk | `freeze` / snap-aware polish で head 忠実性は改善可能 |
| 論文 | 著者取り下げ。同型 DNN-EML の他論文・再現なし |

**ギャップ**: 単一 seed・式別チューニング前提の成功はあるが、**固定プロトコル + 多シード** で「まとも」と言えるベースラインが未確立。

---

## 2. 「まとも」の定義（本フェーズの成功基準）

### 2.1 評価スイート（ゲート用）

Phase 3 の 6 ターゲットを使用（`targets.py` 既存）:

| ID | 式 | 閾値モード |
|----|-----|-----------|
| square | `x0^2` | relaxed (5e-2) |
| product | `x0*x1` | relaxed |
| sum | `x0+x1` | strict (1e-2) |
| exp | `exp(x0)` | relaxed |
| sin | `sin(x0)` | relaxed |
| sin_plus | `sin(x0)+x1` | relaxed |

データ: ノイズ 1%（`noise_std_rel=0.01`）、holdout 分割は `sr_eval.py` 既存慣習に合わせる。

### 2.2 ゲート条件（TSTL 着手の Exit Criteria）

**Gate A — 安定コア（必須）**  
対象: `square`, `product`, `sum`, `exp`, `sin`（5 式）  
シード: `{0, 1, 7, 42, 123}`（5 seeds）  
各 (target, seed) で:

- 学習が **有限**（holdout 予測に NaN/Inf なし）
- **symbolic OK** = snapped holdout MSE ≤ 閾値（`SR検証計画.md` の新定義）

合格: **各ターゲットで symbolic OK ≥ 4/5 seeds（80%）**

**Gate B — 挑戦ターゲット（記録必須・合格は任意）**  
対象: `sin_plus`  
同上 5 seeds で成功率を集計。  
- **合格目安**: ≥ 3/5（60%）  
- 未達でも Gate A 達成 + 多シード統計の報告があれば TSTL 着手可（脆さを既知の制約として記録）

**Gate C — スナップ忠実性（必須）**  
Gate A で合格した run について:

- `snap_degrade` = snapped MSE / soft MSE の中央値 **≤ 1.5**（snap-aware polish 既定で ≈1.0 を期待）

**Gate D — 再現報告（必須・ドキュメント）**  
`NSN/texts/再現報告.md` に以下を記載:

- 論文取り下げの事実
- ゲート結果表（mean±std, per-seed）
- Feynman 負の結果は参考として短く引用（本フェーズの合格条件ではない）

---

## 3. スコープ

### 3.1 本フェーズでやること

| # | 内容 | 新規/既存 |
|---|------|----------|
| 1 | **固定ベースラインプロトコル**の定義と実行スクリプト | 新規スクリプト |
| 2 | **多シードゲート評価**（Gate A–C の自動判定） | 新規スクリプト |
| 3 | **二段階学習オプション**（trunk 主導 → head 固定 polish） | `pipeline.py` 拡張 |
| 4 | **trunk 層アクセス API**（将来 TSTL 用の足場） | `trunk.py` 拡張 |
| 5 | **sin_plus 向け軽量プロトコル**（seed 感度の緩和、再起動 best-of-k は評価のみ） | `trainer` / ゲートスクリプト |
| 6 | **再現報告**ドキュメント | 新規 md |
| 7 | 最小テスト（層凍結 API、ゲート判定ロジック） | `tests/` |

### 3.2 本フェーズでやらないこと

- Feynman 12 式フルベンチの再勝負
- EML 演算子・master formula の理論変更
- `f_prev=parent` の学習モード安定化（export 用途のみ維持）
- TSTL `C(k)` 実装本体（次フェーズ）
- ルート `src/hybrid/` への統合コード
- FPGA / 専用 HW

---

## 4. 技術方針（論文以上の「多少の整備」の中身）

大改造ではなく、再現で有効だった機構を **ゲート評価の既定プロトコル** に昇格させる。

### 4.1 ベースラインプロトコル `BaselineProtocol`（案）

```
Stage SEARCH (60%):  full model 学習（現行 Odrzywołek パイプライン）
Stage HARDEN (30%):  同上
Stage POLISH (10%):  snap_aware_polish=True（既定維持）

オプション A（新規・既定 OFF）: trunk_first
  SEARCH: trunk のみ更新（head + 葉ロジット凍結）
  HARDEN/POLISH: head のみ更新（trunk 凍結）+ snap_aware_polish

オプション B（既存フラグのゲート既定化）: freeze_trunk_after_search
  SEARCH 終了後に trunk 凍結 → head が記号負荷を担う
```

**方針**: Gate A 達成のため、まず **現行フル + 式別 HP（`trainer._config_for_target`）** で多シード計測。未達ターゲットのみ `trunk_first` / `freeze` を試す **フォールバック表** をプロトコルに明記（自動グリッドは Phase 1 では手動表で十分）。

### 4.2 trunk 層 API（TSTL 足場）

`MLPTrunk` に追加予定（名前は実装時確定）:

- `named_linear_layers() -> list[nn.Linear]` — 層インデックス付き
- `set_trainable_layers(indices: set[int] | None)` — `None` = 全層、`{1}` = 中間層のみ、等
- `freeze_all()` / `unfreeze_all()`

現行 `num_layers=3` では実質 **2 つの隠れ Linear + 1 出力 Linear** を層として扱う（ReLU はパラメータなしのため TSTL 論文の「層」とはインデックス対応を文書化）。

### 4.3 sin_plus 対策（最小）

- ゲート評価では **5 seeds 集計** を必須とし、単一 seed 不合格を失敗扱いにしない
- プロトコルに **best-of-3 restarts（評価報告用）** をオプション追加（学習アルゴリズム変更ではなくレポート指標）
- `trainer._config_for_target` の `sin_plus` HP は **変更前にベースライン再計測**（現設定を baseline_v0 として固定）

### 4.4 式別 HP の扱い

| 立場 | 内容 |
|------|------|
| ゲート評価 | `_config_for_target` を **許可**（再現済みの現実的設定） |
| 公平性 | プロトコル JSON に HP を全 run 記録し再現可能にする |
| 将来 TSTL | 同一 HP 表の上で Only Bk を比較（HP チューニングと層選択を混同しない） |

---

## 5. 実装ステップ（承諾後の作業順）

### Step 1: ゲート評価ハーネス（優先）

- **新規** `NSN/scripts/baseline_gate_eval.py`
  - `--seeds 0 1 7 42 123`
  - `--targets`（default: Phase 3 全 6）
  - Gate A/B/C 自動判定
  - 出力: `NSN/results/baseline_gate_<timestamp>/summary.{json,md}`
- **新規** `NSN/tests/test_baseline_gate.py`（判定ロジックの単体テスト）

**完了条件**: 現行コードのみで Gate 計測が回り、未達箇所が定量的に見える。

### Step 2: trunk 層 API

- **変更** `NSN/src/trunk.py` — 層凍結ヘルパ
- **新規** `NSN/tests/test_trunk_layers.py`

**完了条件**: 「層 1 のみ `requires_grad=True`」がテストで検証できる。

### Step 3: 二段階学習オプション

- **変更** `NSN/src/pipeline.py` — `OdrzywolekPipelineConfig` に `trunk_only_search: bool`（仮名）
- **変更** `NSN/scripts/baseline_gate_eval.py` — フォールバックプロトコル切替
- 既存 `head_capacity_eval.py` との重複を避け、ゲート用は `baseline_gate_eval` に集約

**完了条件**: `sum` または `sin` で freeze / trunk_first のいずれかが Gate A を改善するか否かがデータで分かる。

### Step 4: ゲート達成までのイテレーション

- Step 1 結果に基づき、**スコープ内** のプロトコル調整のみ（新演算子追加禁止）
- 目標: **Gate A + Gate C 必達**

**完了条件**: `summary.md` に Gate PASS と明記。

### Step 5: 再現報告

- **新規** `NSN/texts/再現報告.md`
- **追記** `NSN/texts/SR検証計画.md` — ベースラインゲート節へのリンク
- **追記** `NSN/daily_report.md`

**完了条件**: 第三者がゲート結果と TSTL 着手判断を読める。

---

## 6. ファイル構成（予定）

```
NSN/
├── src/
│   ├── trunk.py              # 変更: 層凍結 API
│   └── pipeline.py           # 変更: trunk_only_search 等
├── scripts/
│   └── baseline_gate_eval.py # 新規: 多シードゲート
├── tests/
│   ├── test_trunk_layers.py  # 新規
│   └── test_baseline_gate.py # 新規
├── texts/
│   ├── NSN整備計画_TSTL統合前.md  # 本ファイル
│   └── 再現報告.md           # Step 5 で新規
└── results/
    └── baseline_gate_<ts>/   # ゲート出力
```

---

## 7. リスクと緩和

| リスク | 緩和 |
|--------|------|
| Gate A が現行のまま未達 | `freeze_trunk_after_search` / `trunk_first` をフォールバック。それでも未達なら Gate 定義の見直しをユーザーと合意 |
| sin_plus が恒常的に不安定 | Gate B は任意。ベースライン報告に成功率のみ記載し TSTL は Gate A 通過で着手 |
| 式別 HP で「チューニングしすぎ」批判 | 全 HP を results JSON に記録。TSTL フェーズは同一 HP で層のみ変える |
| 論文取り下げ | 再現報告で明示。研究対象は「v1 提案の再現 + TSTL 拡張」 |

---

## 8. 工数目安

| Step | 目安 |
|------|------|
| 1 ゲートハーネス | 0.5–1 日 |
| 2 trunk API | 0.5 日 |
| 3 二段階学習 | 0.5–1 日 |
| 4 ゲート達成イテレーション | 0.5–1 日（計測時間含む） |
| 5 再現報告 | 0.5 日 |
| **合計** | **約 2–4 日** |

---

## 9. TSTL 統合への引き継ぎ

本フェーズ Exit（Gate A + C + 再現報告）後:

1. `TSTL/texts/作業計画書.md` 作成（`C(k)` 最小実装）
2. NSN trunk 層スキャン `scripts/trunk_layer_profile.py`（Step 2 API 利用）
3. Only Bk / Boost Bk vs full trunk 比較

NSN 整備で確立した **同一ベースラインプロトコル** を TSTL 実験でも使用する。

---

## 10. 承諾（ユーザー記入）

- [x] **§2 ゲート定義**（Gate A: 5 式 × 5 seed で 80%、Gate B: sin_plus 任意）— 承諾
- [x] **§3 スコープ**（Feynman 再勝負なし、TSTL 本体は次フェーズ）— 承諾
- [x] **§4 技術方針**（式別 HP 許可、trunk_first / freeze フォールバック）— 承諾
- [x] **§5 実装ステップの順序** — 承諾

**承諾日**: 2026-07-13  
**補足**: ルート `data/FeynmanEquations.csv` を配置済み（本フェーズのゲート外）。

---

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-07-13 | 初版作成（TSTL 統合前ベースライン整備計画、承諾待ち） |
| 2026-07-13 | §10 ユーザー承諾。Step 1–3 実装（baseline_gate, trunk 層 API, trunk_only_search） |
| 2026-07-13 | Step 4–5: ゲート本番実行（full/freeze/trunk_first）。Gate A FAIL、Gate C PASS（full/freeze）。`再現報告.md` 作成。TSTL 着手可（条件付き）。 |
