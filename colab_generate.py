#!/usr/bin/env python3
"""
Irodori VoiceDesign で captions.json の全ペルソナ分の音声を一括生成する。

前提:
  - Aratako/Irodori-TTS を clone 済み（--irodori-dir で場所を渡す）
  - その中で `uv sync --extra cu128` 済み（GPU 必須。Colab の GPU ランタイム推奨）
  - CLI は Irodori-TTS の README に準拠:
      uv run --no-sync python infer.py \
        --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
        --text "<読み上げ文>" --caption "<声の説明>" --no-ref \
        --output-wav <出力先.wav>

出力:
  <outdir>/voice_01.wav ... voice_50.wav
  <outdir>/captions.json   （viewer.html にそのままドロップできるよう同梱）
  voices.zip               （まとめてダウンロード用。--zip 指定時）

注意:
  - infer.py を1件ずつ呼ぶため、モデルは各回ロードされます（50件だと相応に時間がかかる）。
    速度優先で最適化したい場合は Python API でモデルを1度だけ載せてループする実装に差し替える
    （そのときは Irodori 側の API を一次で確認してから）。
  - 読み上げ文(--text)は「中立文」を既定にしています。将来この声を動画等に使うとき、
    デモ文に性能主張・宣伝が焼き込まれて後で矛盾しないようにするためです。
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# 声質を聞き分けやすく、かつ性能主張・宣伝を含まない中立の読み上げ文
DEFAULT_TEXT = "こんにちは、はじめまして。私の声はこんな感じです。これから少しずつ、いろいろなお話をしていけたら嬉しいです。"
DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"


def load_captions(path: Path):
    """captions.json を [{file, caption}, ...] に正規化して返す。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    if isinstance(data, list):
        for i, o in enumerate(data, 1):
            f = o.get("file") or o.get("name") or o.get("filename") or f"voice_{i:02d}.wav"
            items.append({"file": Path(f).name, "caption": o.get("caption") or o.get("text") or ""})
    elif isinstance(data, dict):
        for i, (k, v) in enumerate(data.items(), 1):
            cap = v if isinstance(v, str) else (v.get("caption") or v.get("text") or "")
            items.append({"file": Path(k).name, "caption": cap})
    else:
        raise ValueError("captions.json は配列 or オブジェクトで渡してください")
    return items


def main():
    ap = argparse.ArgumentParser(description="Irodori VoiceDesign で50ペルソナ音声を一括生成")
    ap.add_argument("--captions", default="captions.json", help="ペルソナ定義 JSON")
    ap.add_argument("--irodori-dir", default="Irodori-TTS", help="clone した Irodori-TTS のパス")
    ap.add_argument("--outdir", default="outputs", help="出力フォルダ")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="読み上げ文（中立文が既定）")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HuggingFace チェックポイント")
    ap.add_argument("--zip", action="store_true", help="生成後に voices.zip へまとめる")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 件だけ生成（動作確認用。0=全件）")
    args = ap.parse_args()

    cap_path = Path(args.captions).resolve()
    irodori = Path(args.irodori_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not (irodori / "infer.py").exists():
        sys.exit(f"[error] {irodori}/infer.py が見つかりません。--irodori-dir を確認してください")

    items = load_captions(cap_path)
    if args.limit:
        items = items[: args.limit]
    print(f"[info] {len(items)} 件を生成します（checkpoint={args.checkpoint}）")
    print(f"[info] 読み上げ文: {args.text}")

    ok, ng = 0, []
    for idx, it in enumerate(items, 1):
        out_wav = outdir / it["file"]
        cmd = [
            "uv", "run", "--no-sync", "python", "infer.py",
            "--hf-checkpoint", args.checkpoint,
            "--text", args.text,
            "--caption", it["caption"],
            "--no-ref",
            "--output-wav", str(out_wav),
        ]
        print(f"[{idx}/{len(items)}] {it['file']}  «{it['caption']}»")
        r = subprocess.run(cmd, cwd=str(irodori))
        if r.returncode == 0 and out_wav.exists():
            ok += 1
        else:
            ng.append(it["file"])
            print(f"  [warn] 生成に失敗: {it['file']}")

    # viewer.html にそのまま渡せるよう captions.json を出力先にも置く
    shutil.copyfile(cap_path, outdir / "captions.json")

    print(f"\n[done] 成功 {ok} / 失敗 {len(ng)}  -> {outdir}")
    if ng:
        print("  失敗:", ", ".join(ng))

    if args.zip:
        zip_base = str(Path("voices").resolve())
        shutil.make_archive(zip_base, "zip", str(outdir))
        print(f"[zip] {zip_base}.zip を作成しました")


if __name__ == "__main__":
    main()
