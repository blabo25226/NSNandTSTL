# TSTL Phase R1 — GPU 実行手順書（Windows）

CPU の Claude Code クラウド環境では **GPU なし・HuggingFace ブロック**のため、実 LLM + GRPO の
層スキャン（R1）は動かせない。本書は **GPU 付き Windows PC** で `scripts/run_layer_scan.py` を
実行するための手順。実行結果は `TSTL/results/tstl_r1_<日時>/` に**自動保存**される。

| 項目 | 内容 |
|------|------|
| 対象 | Windows 10/11 + NVIDIA GPU |
| 実行ファイル | `TSTL/scripts/run_layer_scan.py`（1 コマンドで全パイプライン） |
| ブランチ | `claude/tstl-reproduction-coding-s267ic`（マージ後は既定ブランチ） |
| 結果 | `TSTL/results/tstl_r1_<日時>/`（自動保存、`run.log` 含む） |

---

## 0. 前提

- **NVIDIA GPU + 最新ドライバ**（`nvidia-smi` が動くこと）。
- **VRAM 目安**: R1 `quick`（Qwen2.5-0.5B, bf16）で概ね 6–8GB。`standard`/`full` や R2（1.7B）は増加。
  不足時は §7 参照（`--dtype float16`・プリセット縮小）。
- **Python 3.10+**、**インターネット接続**（Qwen モデルと GSM8K を HuggingFace から取得）。
- Git（リポジトリ取得用）。

---

## 1. リポジトリ取得（PowerShell）

```powershell
cd $HOME
git clone https://github.com/blabo25226/NSNandTSTL.git
cd NSNandTSTL
git checkout claude/tstl-reproduction-coding-s267ic
```

> すでに clone 済みなら `git fetch origin` → `git checkout claude/tstl-reproduction-coding-s267ic` → `git pull`。

---

## 2. 仮想環境の作成

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

`Activate.ps1` が実行ポリシーで拒否される場合（このセッションのみ許可）:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

プロンプト先頭が `(.venv)` になれば OK。

---

## 3. 依存インストール

**CUDA 対応 torch を先に**入れる（自分の CUDA に合わせて `cu121`/`cu124` を選ぶ。確認は `nvidia-smi` 右上の CUDA Version）。

```powershell
python -m pip install --upgrade pip
# 例: CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 残りの依存（bitsandbytes は Windows では自動除外される）
pip install -r TSTL\requirements-r.txt
```

**GPU 認識の確認**（`True` が出ること）:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

`False` の場合は §7 のトラブルシュートへ。

---

## 4. まず dry-run（GPU/ネット不要の事前確認）

構成とコスト見積だけを表示し、環境が壊れていないか確認する。

```powershell
cd TSTL
python scripts\run_layer_scan.py --preset quick --dry-run
```

`out_dir` や `estimated_grpo_generation_passes` などの JSON が表示されれば準備完了。

---

## 5. 本実行

```powershell
# TSTL ディレクトリ内で
python scripts\run_layer_scan.py --preset quick
```

よく使うオプション:

| オプション | 用途 |
|-----------|------|
| `--preset {quick,standard,full}` | 規模。まず `quick`（スモーク） |
| `--dtype {bfloat16,float16,float32}` | 既定 bf16。bf16 非対応 GPU は `float16` |
| `--model <HF名>` | 既定 `Qwen/Qwen2.5-0.5B-Instruct` |
| `--strategy-k <int>` | Only Bk / Mid-k の k（既定 3） |
| `--lr <float>` | 学習率（既定 1e-5） |
| `--out-dir <path>` | 出力先を固定（**中断後の再開に使う**） |

**中断からの再開**: 同じ `--out-dir` を渡すと、`contributions.json` に記録済みの層はスキップして続きから走る（`resume_layer_scan`）。

```powershell
python scripts\run_layer_scan.py --preset quick --out-dir results\my_run
# 中断したら同じコマンドを再実行 -> 残りの層だけ学習
```

パイプライン: `S_base → Full GRPO → 全層（stride 間引き）スキャン → C(k) → Only Bk / Mid-k` の順で自動実行。

---

## 6. 結果（自動保存される中身）

`TSTL\results\tstl_r1_<日時>\`（`--out-dir` 指定時はそのフォルダ）に自動生成:

| ファイル | 内容 |
|----------|------|
| **`report.md`** | 人が読むまとめ（S_base/S_full、C(k) 表、Full vs Only Bk / Mid-k 比較表） |
| **`report.json`** | 上記の機械可読版（config・全スコア・戦略を一括） |
| **`run.log`** | 標準出力の全ログ（進捗・S_base/S_full・各層スコア）。**切断しても残る** |
| `contributions.json` | s_base / s_full / 層ごと S_k / C(k) |
| `contributions_bar.png`, `contributions_depth.png` | C(k) 棒グラフ・深さ正規化プロット |
| `summary.md` | 層スキャンの短いサマリ |
| `strategy_only_bk.json`, `strategy_mid_k.json` | 各戦略の層と S |
| `config.json` | 実行設定 |
| `base_model/`, `full/`, `layer_k/`, `strategy_*/` | モデル/チェックポイント |

> まず `report.md` を見れば全体像が分かる。数値の追跡は `run.log`。

---

## 7. プリセットと所要目安

| preset | train/eval | steps | gen | 用途 |
|--------|-----------|-------|-----|------|
| `quick` | 32 / 16 | 25 | 2 | スモーク（まずこれ） |
| `standard` | 128 / 32 | 80 | 2 | 少し本格的 |
| `full` | 256 / 64 | 200 | 4 | R1 の最大（数時間〜） |

生成回数の目安は dry-run の `estimated_grpo_generation_passes` を参照。**R2（Qwen3-1.7B + NuminaMath）へ
上げる場合**は VRAM 24GB 級・実行数日を見込み、`--model` とデータを差し替える（別途設計）。

---

## 8. トラブルシュート

| 症状 | 対処 |
|------|------|
| `torch.cuda.is_available() == False` | ドライバ更新。torch を CUDA 版で入れ直す（§3 の `--index-url`）。CPU 版 torch が入っていないか確認 |
| `bfloat16` 関連エラー / 古い GPU | `--dtype float16` を付ける |
| VRAM 不足（OOM） | `--preset quick`、`--dtype float16`、より小さい `--model` |
| `Activate.ps1` が実行できない | §2 の `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| モデル/データの取得に失敗 | ネット接続確認。gated モデルは `huggingface-cli login` でトークン設定 |
| trl / transformers の API 差異 | `requirements-r.txt` のバージョンで固定インストール（`build_grpo_trainer` は version-safe に実装済み） |

---

## 9. 結果の持ち帰り（任意）

結果はローカルディスクに残る。リポジトリへ反映するなら手動で:

```powershell
cd $HOME\NSNandTSTL
git add TSTL/results/tstl_r1_<日時>
git commit -m "R1 results (<preset>, <model>)"
git push origin claude/tstl-reproduction-coding-s267ic
```

> 大きなモデル ckpt を含めたくない場合は `report.md` / `report.json` / `contributions*` / `run.log`
> だけを add する。

---

## 10. 実行後にわかること（R1 の判定）

`report.md` で以下を確認（論文再現計画書 §6 の E2/E3/E5 に対応）:

- **E2**: `S_full > S_base`（Full GRPO がベースを改善）。
- **E3**: 全層の C(k) が有限値で図が出る。
- **E5**: `strategy only_bk` の S が Full 以上（単層で全パラメータ GRPO に匹敵）。

`quick` はスモーク規模なので数値は荒い。傾向確認後、`standard`／`full`、さらに R2 へ拡張していく。
