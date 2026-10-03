# irodori-voice-audition

**English** | [日本語](README.ja.md)

> A small toolkit to batch-generate many persona voices with Irodori-TTS VoiceDesign, then audition them side-by-side with captions to pick a voice for your avatar.

Made for the case where you don't yet have a concrete image of the voice you want. Instead of putting the target into words first, you generate a wide spread of candidates, listen through them, and let your ear find the direction.

## Flow

```
[1] captions.json (voice descriptions for many personas)
        |  <- (re)generate with prompts/persona-captions-prompt.md
        v
[2] Generate voice_01..NN.wav with Irodori VoiceDesign on Colab (GPU)
        |  <- batch_gen.py (recommended, fast) / colab_generate.py (simple)
        v
[3] Open outputs/ in viewer.html, audition with captions, pick your voice
```

You can also generate the same captions with several Irodori models and audition them in a **caption x model** grid (`batch_gen.py --models` -> `compare.html`).

## What's inside

| File | Role |
|---|---|
| `captions.json` | Voice descriptions for 50 personas (ready-to-use starter set) |
| `captions_emotions.json` | Test set for comparing how emotion captions land (female/male x 10 emotions, same read-aloud text for all) |
| `prompts/persona-captions-prompt.md` | Prompt for an LLM (Claude Code / Codex, etc.) to (re)generate `captions.json` |
| `batch_gen.py` | **Recommended, fast.** Loads the model once and generates all voices (50 in ~107s on a Colab L4) |
| `colab_generate.py` | Simple fallback. Calls `infer.py` once per caption (slow) |
| `viewer.html` | Audition the results as a **caption + play button** list (review only, no generation, dependency-free single HTML) |
| `compare.html` | Audition multi-model output in a grid with **captions as rows and models as columns** (dependency-free single HTML) |

## Requirements

- Python 3.10+ and a GPU (needed for Irodori VoiceDesign inference; **a Google Colab GPU runtime is recommended**)
- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (default `Aratako/Irodori-TTS-600M-v3-VoiceDesign`; see [2'] for the other supported models)
- `viewer.html` runs in any modern browser (a local server is optional)

## [1] Prepare captions

The starter `captions.json` works as-is. To regenerate it or change the count, hand
`prompts/persona-captions-prompt.md` to an LLM (the more you vary gender, age, timbre, speaking style and mood, the more likely you are to stumble on a good voice while auditioning).

## [2] Batch-generate on Colab (GPU runtime)

Select a **GPU runtime** on Colab and run the cells in order. **`batch_gen.py` (loads the model once) is recommended.**
Measured (Colab L4): dependency sync a few minutes -> 50 voices in **~107s** (one model load + ~2s per item).

```bash
# 0) Get this repo (for captions.json / batch_gen.py)
!git clone https://github.com/shinshin86/irodori-voice-audition.git
%cd irodori-voice-audition

# 1) Set up Irodori-TTS (GPU required)
!git clone --depth 1 https://github.com/Aratako/Irodori-TTS.git
!pip -q install uv
!cd Irodori-TTS && uv sync --extra cu128

# 2) Copy batch_gen.py next to irodori_tts so it can import it
!cp batch_gen.py Irodori-TTS/

# 3) Generate (3 items first as a smoke test, then all)
!cd Irodori-TTS && uv run --no-sync python batch_gen.py --captions ../captions.json --outdir ../outputs --limit 3
!cd Irodori-TTS && uv run --no-sync python batch_gen.py --captions ../captions.json --outdir ../outputs

# 4) Zip and download
import shutil; shutil.make_archive('voices','zip','outputs')
from google.colab import files; files.download('voices.zip')
```

- **The read-aloud text (`--text`) defaults to a neutral sentence.** If you later reuse a voice in a video, this avoids baking a claim or promo line into the demo audio that could contradict things afterward. Override with `--text "..."`.
- **Reference-free generation from the caption alone (`--no-ref`)** — pure VoiceDesign. All parameters match the defaults in the official `infer.py` argparse (`cfg 3.0/3.0`, `guidance=independent`, etc.; `num_steps` follows the checkpoint default: 40 for regular models, 4 for MeanFlow models).
- If an entry in captions.json has a `"text"` field, that entry uses it as its read-aloud text.
- **Resume-friendly**: existing wavs are skipped, so a re-run continues where it stopped. Pointing it at a folder made by a different model stops with an error, so voices from two models never mix.
- The generation API is the official `infer.py` (`InferenceRuntime` / `SamplingRequest`). To use the CLI directly:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "<read-aloud text>" --caption "<voice description>" --no-ref \
  --output-wav outputs/voice_01.wav
```

### Simple `colab_generate.py`
A fallback for environments where `batch_gen.py` doesn't work. It calls `infer.py` once per caption, reloading the model each time, so 50 items are slow (tens of seconds each). Output is the same.

## [2'] Generate with several models (`--models`)

Pass a comma-separated list to `--models` and the script loads each model in turn and generates the same captions. Output goes to `outputs/<model>/`, and the list is written to `outputs/models.json`.

```bash
!cd Irodori-TTS && uv run --no-sync python batch_gen.py \
    --captions ../captions_emotions.json --outdir ../outputs-emotions \
    --models v3,v4.1,v4.1-mf,v4-large --seed 0
```

| Alias | Checkpoint | Notes |
|---|---|---|
| `v2` | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | Older release |
| `v3` | `Aratako/Irodori-TTS-600M-v3-VoiceDesign` | Default when `--models` is not given |
| `v4` | `Aratako/Irodori-TTS-v4-Small` | The author recommends v4.1 instead |
| `v4.1` | `Aratako/Irodori-TTS-v4.1-Small` | |
| `v4.1-mf` | `Aratako/Irodori-TTS-v4.1-Small-MF` | MeanFlow distilled, 4 steps (CFG settings don't apply) |
| `v4.1-int8` | `Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only` | torchao quantized (bf16 automatically) |
| `v4-large` | `Aratako/Irodori-TTS-v4-Large` | 3.29B. **Subject to the Gemma Terms of Use** |
| `v4-large-int8` | `Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only` | Quantized Large (bf16 automatically). **Subject to the Gemma Terms of Use** |

- Besides aliases, you can pass a Hugging Face repo id (`owner/repo` or `owner/repo/subfolder`).
- `--precision auto` (default) loads quantized checkpoints in bf16 and everything else in fp32. Use `--precision bf16` if you run out of memory.
- Fix `--seed` to make a run reproducible (the seed actually used is recorded in `models.json`).
- If one model fails to load (e.g. out of VRAM), the remaining models still run. Re-running skips wavs that already exist.
- **Use a separate `--outdir` per caption set** (`models.json` assumes a single caption set).

## [3] Audition

Extract the zip you downloaded from Colab into **this folder's `outputs/`** (`outputs/voice_*.wav` + `outputs/captions.json`),
then start a simple server here and open `viewer.html`. It **auto-loads `outputs/`** and lists everything.

```bash
python3 -m http.server 8000
# Open http://localhost:8000/viewer.html in a browser -> outputs/ is shown automatically
```

Each voice appears with its caption, so you can play through and pick the one you like. Everything runs locally in the browser; the audio is never uploaded anywhere.
(If you open `viewer.html` directly without a server, just drop the audio + `captions.json` onto the page to get the same view.)

- **Which model made it**: above the list, the page shows the model that generated the voices, e.g. "生成モデル: v3 (Aratako/Irodori-TTS-600M-v3-VoiceDesign)". It reads the `generation.json` that `batch_gen.py` writes into the output folder. Older output without this record shows "記録なし" (no record); if you know the model, drop in a `generation.json` such as `{"id": "v3", "checkpoint": "Aratako/Irodori-TTS-600M-v3-VoiceDesign"}`.
- **Multi-model output** (`--models`) gets tabs at the top to switch between models, one list per model. Use `viewer.html?dir=outputs-emotions` for a folder with another name, and add `&model=v4.1` to open a specific model.

## [3'] Compare models (`compare.html`)

Put the output folder from `--models` in this repo, start a server and open `compare.html` (it reads `outputs/` by default; pass `?dir=` for a folder with another name).

```bash
python3 -m http.server 8000
# http://localhost:8000/compare.html -> outputs/models.json is shown automatically
# For another folder: http://localhost:8000/compare.html?dir=outputs-emotions
```

- Captions run down the rows and models across the columns. Press ▶ in a cell to play that combination (the clip length is shown on the right).
- **Listen across**: "▶ 横に再生" (play across) on a row plays the same caption through every model in turn, so you hear how each model renders the same description.
- **Listen down**: "▼ 縦に再生" (play down) on a column header plays one model through every caption from top to bottom, so you hear how the voice changes as the caption (e.g. the emotion) changes.
- If captions have a `tag` (such as the emotions in `captions_emotions.json`), the buttons above the grid filter to that tag.
- Keyboard: `←→` model, `↑↓` caption (moving plays that cell), `Space` play/stop, `R` play the current row across, `C` play the current column down.
- Without a server, drop the output folder (containing `models.json`) onto the page or use "出力フォルダを選ぶ" (choose output folder).

## Notes

- Generated artifacts (`outputs/`, wav, zip) are not committed to the repo (already in `.gitignore`).
- **Follow the upstream license and terms** for the Irodori-TTS model and codec. Whether you may use the generated audio also depends on the upstream terms.
- In particular, `v4-large` / `v4-large-int8` use a Gemma-derived text encoder and are subject to the **Gemma Terms of Use and Prohibited Use Policy** (see each model card).

## Acknowledgments

- Voice generation uses the VoiceDesign model from [**Irodori-TTS**](https://github.com/Aratako/Irodori-TTS) by [@Aratako](https://github.com/Aratako). Thanks for releasing such a great model.
- This repository is a wrapper/viewer that batch-generates using the inference flow of Irodori-TTS's official `infer.py`.

## License

MIT License (see `LICENSE`). As noted above, the Irodori-TTS model itself and any generated audio remain subject to the upstream license and terms.
