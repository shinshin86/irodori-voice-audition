# irodori-voice-audition

[English](README.md) | **日本語**

> Irodori-TTS VoiceDesign で多数のペルソナ声を一括生成し、キャプション付きで聞き比べて「アバターの声」を探すための小さなツール群。

「どんな声にするか具体的なイメージが無い」状態から、たくさんの声を並べて耳で当たりに寄せていくための道具立てです。声のイメージを先に言葉にできなくても、幅を持たせた候補を一気に聴いて方向性を掴む、という探し方に向いています。

## 流れ

```
[1] captions.json（多数ペルソナの「声の説明」）
        │  ← prompts/persona-captions-prompt.md で(再)生成できる
        ▼
[2] Colab(GPU) で Irodori VoiceDesign を回して voice_01..NN.wav を生成
        │  ← batch_gen.py（推奨・高速）/ colab_generate.py（簡易）
        ▼
[3] outputs/ を viewer.html で開いてキャプション付きで聞き比べ、好みの声を選ぶ
```

複数の Irodori モデルで同じキャプションを生成し、**どのモデルが狙った声・感情を一番よく出すか**を採点しながら比べることもできます（`batch_gen.py --models` → `compare.html`）。

## 中身

| ファイル | 役割 |
|---|---|
| `captions.json` | 50ペルソナの「声の説明」（すぐ使える初期セット） |
| `captions_emotions.json` | 感情キャプションの当たり方を比べる検証セット（女性・男性 × 10感情。読み上げ文は全件共通） |
| `prompts/persona-captions-prompt.md` | captions.json を(再)生成する LLM（Claude Code / Codex 等）用プロンプト |
| `batch_gen.py` | **推奨・高速版**。モデル1回ロードで全件生成（Colab L4 で50件 約107秒） |
| `colab_generate.py` | 簡易版。`infer.py` を1件ずつ呼ぶフォールバック（遅い） |
| `viewer.html` | 生成結果を **キャプション＋再生ボタン** で並べて聞き比べる（生成はしない・確認専用・依存なしの単一HTML） |
| `compare.html` | 複数モデルの出力を **キャプション × モデル** の表で聞き比べ、5段階で採点・集計する（依存なしの単一HTML） |

## 必要なもの

- Python 3.10+ と GPU（Irodori VoiceDesign の推論に必要。**Google Colab の GPU ランタイム推奨**）
- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（既定は `Aratako/Irodori-TTS-600M-v3-VoiceDesign`。ほかの対応モデルは [2'] を参照）
- `viewer.html` はモダンブラウザだけで動く（サーバー不要でも可）

## [1] キャプションを用意

初期セット `captions.json` がそのまま使えます。作り直したい/件数を変えたいときは
`prompts/persona-captions-prompt.md` を LLM に渡して再生成してください（性別・年齢・声質・話し方・雰囲気の軸を散らすほど、聞き比べで良い声に出会いやすい）。

## [2] Colab で一括生成（GPUランタイム）

Colab で **GPU ランタイム**を選び、順に実行。**`batch_gen.py`（モデル1回ロードの高速版）を推奨**。
実測（Colab L4）: 依存同期 数分 → 50件生成 **約107秒**（1回ロード＋1件約2秒）。

```bash
# 0) このリポジトリを取得（captions.json / batch_gen.py を使う）
!git clone https://github.com/shinshin86/irodori-voice-audition.git
%cd irodori-voice-audition

# 1) Irodori-TTS を用意（GPU 必須）
!git clone --depth 1 https://github.com/Aratako/Irodori-TTS.git
!pip -q install uv
!cd Irodori-TTS && uv sync --extra cu128

# 2) batch_gen.py を Irodori-TTS 直下へ（irodori_tts を import するため）
!cp batch_gen.py Irodori-TTS/

# 3) 一括生成（まず 3 件で動作確認 → 全件）
!cd Irodori-TTS && uv run --no-sync python batch_gen.py --captions ../captions.json --outdir ../outputs --limit 3
!cd Irodori-TTS && uv run --no-sync python batch_gen.py --captions ../captions.json --outdir ../outputs

# 4) zip してダウンロード
import shutil; shutil.make_archive('voices','zip','outputs')
from google.colab import files; files.download('voices.zip')
```

- **読み上げ文（`--text`）は中立文が既定**。将来この声を動画等で使うとき、デモ文に性能主張や宣伝が焼き込まれて後から矛盾しないようにするため。変えたいときは `--text "..."`。
- **参照音声なし（`--no-ref`）でキャプションだけから作る** VoiceDesign 純粋生成。全パラメータは公式 `infer.py` の argparse 既定値に一致（`cfg 3.0/3.0`, `guidance=independent` 等。`num_steps` はチェックポイント既定で、通常モデルは 40、MeanFlow モデルは 4）。
- captions.json の各要素に `"text"` があれば、その件だけ読み上げ文を差し替えます。
- **resume 対応**: 既存の wav はスキップするので、中断しても再実行で続きから。
- 生成 API の実体は公式 `infer.py`（`InferenceRuntime` / `SamplingRequest`）。CLI を直接使う場合:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "<読み上げ文>" --caption "<声の説明>" --no-ref \
  --output-wav outputs/voice_01.wav
```

### 簡易版 `colab_generate.py`
`batch_gen.py` が使えない環境向けのフォールバック。`infer.py` を1件ずつ呼ぶためモデルを毎回ロードし、50件だと遅い（1件あたり数十秒）。出力は同じ。

## [2'] 複数モデルで比較生成（`--models`）

`--models` にモデルをカンマ区切りで渡すと、1モデルずつロードして同じキャプションを順に生成します。出力は `outputs/<モデル名>/` に分かれ、一覧が `outputs/models.json` に書かれます。

```bash
!cd Irodori-TTS && uv run --no-sync python batch_gen.py \
    --captions ../captions_emotions.json --outdir ../outputs-emotions \
    --models v3,v4.1,v4.1-mf,v4-large --seed 0
```

| 短縮名 | チェックポイント | 備考 |
|---|---|---|
| `v2` | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | 旧版 |
| `v3` | `Aratako/Irodori-TTS-600M-v3-VoiceDesign` | `--models` 未指定時の既定 |
| `v4` | `Aratako/Irodori-TTS-v4-Small` | 作者は v4.1 を推奨 |
| `v4.1` | `Aratako/Irodori-TTS-v4.1-Small` | |
| `v4.1-mf` | `Aratako/Irodori-TTS-v4.1-Small-MF` | MeanFlow 蒸留版。4ステップで速い（CFG 設定は効かない） |
| `v4.1-int8` | `Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only` | torchao 量子化版（自動で bf16） |
| `v4-large` | `Aratako/Irodori-TTS-v4-Large` | 3.29B。**Gemma 利用規約の対象** |
| `v4-large-int8` | `Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only` | Large の量子化版（自動で bf16）。**Gemma 利用規約の対象** |

- 短縮名のほか、HF の repo id（`owner/repo` や `owner/repo/subfolder`）もそのまま指定できます。
- `--precision auto`（既定）は量子化版だけ bf16、それ以外は fp32 で読み込みます。メモリが足りないときは `--precision bf16`。
- `--seed` を固定すると、あとで同じ条件で作り直せます（実際に使った seed は `models.json` に記録されます）。
- あるモデルのロードに失敗しても（VRAM 不足など）、残りのモデルは続けて生成します。再実行すると生成済みの wav は飛ばします。
- **キャプションセットごとに `--outdir` を分けてください**（`models.json` はキャプションセット1つ分を前提にしています）。

## [3] 聞き比べ

Colab から落とした zip を **このフォルダの `outputs/` に展開**（`outputs/voice_*.wav` ＋ `outputs/captions.json`）したら、
このフォルダで簡易サーバーを立てて `viewer.html` を開くだけ。**`outputs/` を自動で読み込んで一覧表示**します。

```bash
python3 -m http.server 8000
# ブラウザで http://localhost:8000/viewer.html を開く → outputs/ を自動表示
```

各声がキャプション付きで並ぶので、再生して好みの声を探します。すべてローカル・ブラウザ内で完結し、音声はどこにも送信されません。
（サーバーを立てずに `viewer.html` を直接開いた場合は、音声＋`captions.json` を画面にドロップすれば同じように表示できます）

## [3'] モデル比較（`compare.html`）

`--models` で生成した出力フォルダを `outputs/` に置いて（または `outputs/` へリネームして）サーバーを立て、`compare.html` を開きます。

```bash
python3 -m http.server 8000
# http://localhost:8000/compare.html → outputs/models.json を自動表示
```

- 各キャプションの下にモデルが横並びになり、それぞれ再生して **1〜5 で採点**、行ごとに **★ でベストを1つ**選べます。「▶ 順に再生」で同じキャプションをモデル順に続けて聴けます。
- 上の集計表に、モデルごとの平均スコアと★の数が **感情（`tag`）別** に出ます。列ごとの最高値は枠で強調されるので、「怒りなら v4-large、照れなら v4.1」のような傾向を拾えます。
- **ブラインド**: モデル名を「モデル A/B/C…」に伏せ、行ごとに並び順をシャッフルします。先入観なしに採点したいときに使います（ブラインド中は集計を隠します）。
- キーボード操作: `↑↓` 行、`←→` モデル（移動すると再生）、`Space` 再生/停止、`1`〜`5` 採点、`B` ベスト、`R` 行を順に再生、`N` 次の未採点へ。
- 採点はブラウザ内（localStorage）に保存されます。JSON / CSV で書き出せて、JSON は読み込んで続きから再開できます。
- サーバーを立てずに開いた場合は、出力フォルダ（`models.json` を含む）をドロップするか「outputs フォルダを選ぶ」から選択してください。

## 注意

- 生成物（`outputs/`・wav・zip）はリポジトリに含めません（`.gitignore` 済み）。
- Irodori-TTS 側のモデル・コーデックの**ライセンス・利用条件は本家に従ってください**。生成した音声の利用可否も本家の規約に準じます。
- 特に `v4-large` / `v4-large-int8` は、テキストエンコーダが Gemma 由来のため **Gemma 利用規約と禁止用途ポリシー**の対象です（各モデルカード参照）。

## Acknowledgments

- 音声生成は [**Irodori-TTS**](https://github.com/Aratako/Irodori-TTS)（[@Aratako](https://github.com/Aratako)）の VoiceDesign モデルを利用しています。素晴らしいモデルの公開に感謝します。
- 本リポジトリは Irodori-TTS の公式 `infer.py` の推論手順に沿って一括生成を行うラッパー／ビューアです。

## License

MIT License（`LICENSE` を参照）。ただし上記のとおり、Irodori-TTS 本体および生成音声の扱いは本家のライセンス・規約に従ってください。
