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

## What's inside

| File | Role |
|---|---|
| `captions.json` | Voice descriptions for 50 personas (ready-to-use starter set) |
| `prompts/persona-captions-prompt.md` | Prompt for an LLM (Claude Code / Codex, etc.) to (re)generate `captions.json` |
| `batch_gen.py` | **Recommended, fast.** Loads the model once and generates all voices (50 in ~107s on a Colab L4) |
| `colab_generate.py` | Simple fallback. Calls `infer.py` once per caption (slow) |
| `viewer.html` | Audition the results as a **caption + play button** list (review only, no generation, dependency-free single HTML) |

## Requirements

- Python 3.10+ and a GPU (needed for Irodori VoiceDesign inference; **a Google Colab GPU runtime is recommended**)
- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (`Aratako/Irodori-TTS-600M-v3-VoiceDesign`)
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
- **Reference-free generation from the caption alone (`--no-ref`)** — pure VoiceDesign. All parameters match the defaults in the official `infer.py` argparse (`num_steps=40`, `cfg 3.0/3.0`, `guidance=independent`, etc.).
- **Resume-friendly**: existing wavs are skipped, so a re-run continues where it stopped.
- The generation API is the official `infer.py` (`InferenceRuntime` / `SamplingRequest`). To use the CLI directly:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "<read-aloud text>" --caption "<voice description>" --no-ref \
  --output-wav outputs/voice_01.wav
```

### Simple `colab_generate.py`
A fallback for environments where `batch_gen.py` doesn't work. It calls `infer.py` once per caption, reloading the model each time, so 50 items are slow (tens of seconds each). Output is the same.

## [3] Audition

Extract the zip you downloaded from Colab into **this folder's `outputs/`** (`outputs/voice_*.wav` + `outputs/captions.json`),
then start a simple server here and open `viewer.html`. It **auto-loads `outputs/`** and lists everything.

```bash
python3 -m http.server 8000
# Open http://localhost:8000/viewer.html in a browser -> outputs/ is shown automatically
```

Each voice appears with its caption, so you can play through and pick the one you like. Everything runs locally in the browser; the audio is never uploaded anywhere.
(If you open `viewer.html` directly without a server, just drop the audio + `captions.json` onto the page to get the same view.)

## Notes

- Generated artifacts (`outputs/`, wav, zip) are not committed to the repo (already in `.gitignore`).
- **Follow the upstream license and terms** for the Irodori-TTS model and codec. Whether you may use the generated audio also depends on the upstream terms.

## Acknowledgments

- Voice generation uses the VoiceDesign model from [**Irodori-TTS**](https://github.com/Aratako/Irodori-TTS) by [@Aratako](https://github.com/Aratako). Thanks for releasing such a great model.
- This repository is a wrapper/viewer that batch-generates using the inference flow of Irodori-TTS's official `infer.py`.

## License

MIT License (see `LICENSE`). As noted above, the Irodori-TTS model itself and any generated audio remain subject to the upstream license and terms.
