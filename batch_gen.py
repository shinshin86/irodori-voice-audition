#!/usr/bin/env python3
"""
Irodori VoiceDesign で captions.json の全ペルソナを「モデル1回ロード」で一括生成する高速版。
（colab_generate.py は infer.py を1件ずつ呼ぶ簡易版。50件ならこちらが圧倒的に速い）

前提:
  - このファイルを clone 済みの Irodori-TTS リポジトリ直下に置き、その venv で実行する:
        cp batch_gen.py /path/to/Irodori-TTS/
        cd /path/to/Irodori-TTS
        uv run --no-sync python batch_gen.py --captions /path/to/captions.json --outdir /path/to/outputs
  - `uv sync --extra cu128` 済み・GPU 必須（Colab GPU で検証済み: L4, 50件 約107秒）

公式 infer.py の Python API（InferenceRuntime / SamplingRequest）を main() の構築手順どおりに
再現している。全パラメータは infer.py の argparse 既定値に一致（num_steps=40, cfg 3.0/3.0,
guidance=independent, trim_tail など）。--no-ref でキャプションのみから生成する。

読み上げ文(--text)は中立文が既定（将来この声を動画等に使うとき、デモ文に性能主張・宣伝が
焼き込まれて矛盾しないようにするため）。
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    resolve_cfg_scales,
    save_wav,
)

DEFAULT_TEXT = "こんにちは、はじめまして。私の声はこんな感じです。これから少しずつ、いろいろなお話をしていけたら嬉しいです。"
DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


def load_captions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    if isinstance(data, list):
        for i, o in enumerate(data, 1):
            f = o.get("file") or o.get("name") or o.get("filename") or f"voice_{i:02d}.wav"
            items.append({"file": Path(f).name, "caption": o.get("caption") or o.get("text") or ""})
    elif isinstance(data, dict):
        for k, v in data.items():
            cap = v if isinstance(v, str) else (v.get("caption") or v.get("text") or "")
            items.append({"file": Path(k).name, "caption": cap})
    else:
        raise ValueError("captions.json は配列 or オブジェクトで渡してください")
    return items


def main():
    ap = argparse.ArgumentParser(description="Irodori VoiceDesign 50ペルソナ一括生成（モデル1回ロード）")
    ap.add_argument("--captions", default="captions.json")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="読み上げ文（中立文が既定）")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 件だけ（動作確認用。0=全件）")
    args = ap.parse_args()

    caps = Path(args.captions).resolve()
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    items = load_captions(caps)
    if args.limit:
        items = items[: args.limit]

    ckpt = hf_hub_download(repo_id=args.checkpoint, filename="model.safetensors")
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=ckpt, model_device=args.device, codec_repo=CODEC_REPO,
        model_precision="fp32", codec_device=args.device, codec_precision="fp32",
        codec_deterministic_encode=True, codec_deterministic_decode=True,
        compile_model=False, compile_dynamic=False,
    ))
    print("=== model loaded ===", flush=True)

    ok, ng, t0 = 0, [], time.time()
    for idx, it in enumerate(items, 1):
        dst = out / it["file"]
        if dst.exists() and dst.stat().st_size > 0:  # resume: 既存はスキップ
            print(f"[{idx}/{len(items)}] skip {it['file']}", flush=True); ok += 1; continue
        cap = it["caption"]
        use_caption = bool(getattr(runtime.model_cfg, "use_caption_condition", True) and cap and cap.strip())
        cs_t, cs_c, cs_s, _ = resolve_cfg_scales(
            cfg_guidance_mode="independent", cfg_scale_text=3.0, cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0, cfg_scale=None,
            use_caption_condition=use_caption, use_speaker_condition=False)
        try:
            res = runtime.synthesize(SamplingRequest(
                text=args.text, caption=cap, ref_wav=None, ref_latent=None, ref_embed=None,
                no_ref=True, ref_normalize_db=-16.0, ref_ensure_max=True,
                num_candidates=1, decode_mode="sequential", seconds=None, duration_scale=1.0,
                max_ref_seconds=30.0, max_text_len=None, max_caption_len=None, num_steps=40,
                cfg_scale_text=cs_t, cfg_scale_caption=cs_c, cfg_scale_speaker=cs_s,
                cfg_guidance_mode="independent", cfg_scale=None, cfg_min_t=0.5, cfg_max_t=1.0,
                truncation_factor=None, rescale_k=None, rescale_sigma=None,
                context_kv_cache=True, speaker_kv_scale=None, speaker_kv_min_t=0.9,
                speaker_kv_max_layers=None, speaker_uncond_mode="mask", seed=None,
                t_schedule_mode="linear", sway_coeff=-1.0, trim_tail=True,
                tail_window_size=20, tail_std_threshold=0.05, tail_mean_threshold=0.1,
                lora_adapter=None), log_fn=None)
            save_wav(str(dst), res.audio, res.sample_rate)
            ok += 1
            print(f"[{idx}/{len(items)}] {it['file']}  ({time.time()-t0:.0f}s累計)", flush=True)
        except Exception as e:
            ng.append(it["file"]); print(f"[{idx}/{len(items)}] FAIL {it['file']}: {e!r}", flush=True)

    shutil.copyfile(caps, out / "captions.json")  # viewer.html にそのまま渡せるよう同梱
    print(f"=== DONE ok={ok} ng={len(ng)} total={time.time()-t0:.0f}s -> {out} ===", flush=True)
    if ng:
        print("failed:", ng, flush=True)


if __name__ == "__main__":
    main()
