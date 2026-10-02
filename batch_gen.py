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

モデルの比較（--models）:
  - `--models v3,v4.1,v4-large` のように複数指定すると、モデルを1つずつロードして順に生成する。
    出力は <outdir>/<モデル名>/voice_XX.wav に分かれ、<outdir>/models.json に一覧が書かれる。
    compare.html でこの models.json を読むと、同じキャプションをモデル横並びで聞き比べられる。
  - 指定できる短縮名は MODEL_ALIASES を参照。HF の repo id（owner/repo または
    owner/repo/subfolder）をそのまま書いてもよい。
  - --models を付けない場合は従来どおり --checkpoint の1モデルを <outdir> 直下に出力する。

公式 infer.py の Python API（InferenceRuntime / SamplingRequest）を main() の構築手順どおりに
再現している。全パラメータは infer.py の argparse 既定値に一致（cfg 3.0/3.0,
guidance=independent, trim_tail など）。num_steps は既定でチェックポイント任せ
（RF モデルは 40、MeanFlow モデルは 4）。--no-ref でキャプションのみから生成する。

読み上げ文(--text)は中立文が既定（将来この声を動画等に使うとき、デモ文に性能主張・宣伝が
焼き込まれて矛盾しないようにするため）。captions.json の各要素に "text" があればそちらを優先する。
"""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    resolve_cfg_scales,
    save_wav,
)

DEFAULT_TEXT = "こんにちは、はじめまして。私の声はこんな感じです。これから少しずつ、いろいろなお話をしていけたら嬉しいです。"
DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"

# VoiceDesign（キャプション指定）に対応したチェックポイントの短縮名
MODEL_ALIASES = {
    "v2": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
    "v3": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
    "v4": "Aratako/Irodori-TTS-v4-Small",
    "v4.1": "Aratako/Irodori-TTS-v4.1-Small",
    "v4.1-mf": "Aratako/Irodori-TTS-v4.1-Small-MF",
    "v4.1-int8": "Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only",
    "v4-large": "Aratako/Irodori-TTS-v4-Large",
    "v4-large-int8": "Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only",
}


def load_captions(path: Path):
    """captions.json を [{file, caption, text?, tag?}, ...] に正規化して返す。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    if isinstance(data, list):
        for i, o in enumerate(data, 1):
            f = o.get("file") or o.get("name") or o.get("filename") or f"voice_{i:02d}.wav"
            it = {"file": Path(f).name, "caption": o.get("caption") or ""}
            for k in ("text", "tag"):
                if o.get(k):
                    it[k] = o[k]
            items.append(it)
    elif isinstance(data, dict):
        for k, v in data.items():
            cap = v if isinstance(v, str) else (v.get("caption") or v.get("text") or "")
            items.append({"file": Path(k).name, "caption": cap})
    else:
        raise ValueError("captions.json は配列 or オブジェクトで渡してください")
    return items


def resolve_model(spec: str):
    """短縮名 or repo id → (出力フォルダ名, repo id)。"""
    spec = spec.strip()
    if spec in MODEL_ALIASES:
        return spec, MODEL_ALIASES[spec]
    if "/" not in spec:
        raise SystemExit(f"[error] 不明なモデル: {spec}（短縮名: {', '.join(MODEL_ALIASES)}）")
    return spec.split("/", 1)[1].replace("/", "_"), spec


def resolve_precision(precision: str, checkpoint: str) -> str:
    if precision != "auto":
        return precision
    # torchao 量子化チェックポイントは bf16 必須（モデルカードの指定）
    return "bf16" if "Quantized" in checkpoint else "fp32"


def generate(items, *, checkpoint, out, args, precision):
    """1モデル分を生成し、ファイルごとの結果を返す。"""
    ckpt = download_hf_checkpoint(checkpoint)
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=ckpt, model_device=args.device, codec_repo=CODEC_REPO,
        model_precision=precision, codec_device=args.device, codec_precision="fp32",
        codec_deterministic_encode=True, codec_deterministic_decode=True,
        compile_model=False, compile_dynamic=False,
    ))
    print(f"=== model loaded: {checkpoint} ({precision}) ===", flush=True)

    files, seeds, ng, t0 = [], {}, [], time.time()
    for idx, it in enumerate(items, 1):
        dst = out / it["file"]
        if dst.exists() and dst.stat().st_size > 0:  # resume: 既存はスキップ
            print(f"[{idx}/{len(items)}] skip {it['file']}", flush=True)
            files.append(it["file"]); continue
        cap = it["caption"]
        use_caption = bool(getattr(runtime.model_cfg, "use_caption_condition", True) and cap and cap.strip())
        cs_t, cs_c, cs_s, _ = resolve_cfg_scales(
            cfg_guidance_mode="independent", cfg_scale_text=3.0, cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0, cfg_scale=None,
            use_caption_condition=use_caption, use_speaker_condition=False)
        try:
            res = runtime.synthesize(SamplingRequest(
                text=it.get("text") or args.text, caption=cap, ref_wav=None, ref_latent=None,
                ref_embed=None, no_ref=True, ref_normalize_db=-16.0, ref_ensure_max=True,
                num_candidates=1, decode_mode="sequential", seconds=None, duration_scale=1.0,
                max_ref_seconds=None, max_text_len=None, max_caption_len=None,
                num_steps=args.num_steps or None,
                cfg_scale_text=cs_t, cfg_scale_caption=cs_c, cfg_scale_speaker=cs_s,
                cfg_guidance_mode="independent", cfg_scale=None, cfg_min_t=0.5, cfg_max_t=1.0,
                truncation_factor=None, rescale_k=None, rescale_sigma=None,
                context_kv_cache=True, speaker_kv_scale=None, speaker_kv_min_t=None,
                speaker_kv_max_layers=None, speaker_uncond_mode="mask", seed=args.seed,
                t_schedule_mode="linear", sway_coeff=-1.0, trim_tail=True,
                tail_window_size=20, tail_std_threshold=0.05, tail_mean_threshold=0.1,
                lora_adapter=None), log_fn=None)
            save_wav(str(dst), res.audio, res.sample_rate)
            files.append(it["file"]); seeds[it["file"]] = res.used_seed
            print(f"[{idx}/{len(items)}] {it['file']}  ({time.time()-t0:.0f}s累計)", flush=True)
        except Exception as e:
            ng.append(it["file"]); print(f"[{idx}/{len(items)}] FAIL {it['file']}: {e!r}", flush=True)

    elapsed = time.time() - t0
    del runtime
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"=== DONE {checkpoint} ok={len(files)} ng={len(ng)} total={elapsed:.0f}s -> {out} ===", flush=True)
    if ng:
        print("failed:", ng, flush=True)
    return {"files": files, "failed": ng, "seeds": seeds, "seconds": round(elapsed, 1)}


def write_manifest(outdir: Path, items, text: str, entry: dict):
    """<outdir>/models.json を更新する（同じ id のモデルは置き換え、他は残す）。"""
    path = outdir / "models.json"
    manifest = {"models": []}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    models = manifest.get("models", [])
    prev = manifest.get("captions") or []
    n = min(len(prev), len(items))
    if prev[:n] != items[:n]:  # --limit で件数だけ違うのは許容
        print(f"[warn] {path} のキャプションと今回の captions が異なります。"
              "キャプションセットごとに --outdir を分けてください", flush=True)
    for i, m in enumerate(models):
        if m["id"] == entry["id"]:  # 同じモデルは並び順を保ったまま置き換え
            entry["seeds"] = {**m.get("seeds", {}), **entry["seeds"]}  # resume 時に過去の seed を残す
            models[i] = entry
            break
    else:
        models.append(entry)
    manifest = {"text": text, "captions": items, "models": models}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Irodori VoiceDesign 一括生成（モデル1回ロード・複数モデル比較対応）")
    ap.add_argument("--captions", default="captions.json")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="読み上げ文（中立文が既定。captions の text が優先）")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="--models 未指定時に使う1モデル")
    ap.add_argument("--models", default="",
                    help=f"比較するモデルをカンマ区切りで（短縮名: {', '.join(MODEL_ALIASES)} / repo id も可）")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16"],
                    help="モデル精度（auto: 量子化版は bf16、それ以外は fp32）")
    ap.add_argument("--num-steps", type=int, default=0, help="サンプリングステップ数（0=チェックポイント既定）")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（固定すると再現できる）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 件だけ（動作確認用。0=全件）")
    args = ap.parse_args()

    caps = Path(args.captions).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    items = load_captions(caps)
    if args.limit:
        items = items[: args.limit]

    if not args.models:
        # 従来どおり: 1モデルを outdir 直下へ。viewer.html にそのまま渡せるよう captions.json を同梱
        generate(items, checkpoint=args.checkpoint, out=outdir, args=args,
                 precision=resolve_precision(args.precision, args.checkpoint))
        shutil.copyfile(caps, outdir / "captions.json")
        return

    models = [resolve_model(s) for s in args.models.split(",") if s.strip()]
    for n, (mid, checkpoint) in enumerate(models, 1):
        print(f"##### [{n}/{len(models)}] {mid} = {checkpoint}", flush=True)
        out = outdir / mid
        out.mkdir(parents=True, exist_ok=True)
        precision = resolve_precision(args.precision, checkpoint)
        try:
            result = generate(items, checkpoint=checkpoint, out=out, args=args, precision=precision)
        except Exception as e:  # ロード失敗（VRAM 不足など）でも残りのモデルは続ける
            print(f"##### FAIL {mid}: {e!r}", flush=True)
            if not any(out.iterdir()):
                out.rmdir()
            continue
        shutil.copyfile(caps, out / "captions.json")
        write_manifest(outdir, items, args.text, {
            "id": mid, "checkpoint": checkpoint, "dir": mid, "precision": precision,
            "num_steps": args.num_steps or "default", **result,
        })
    print(f"=== ALL DONE -> {outdir / 'models.json'}（compare.html で聞き比べ）===", flush=True)


if __name__ == "__main__":
    main()
