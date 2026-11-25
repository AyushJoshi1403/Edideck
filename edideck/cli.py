import argparse, importlib.util, json, os, shutil
from dataclasses import asdict
from pathlib import Path
from . import core


def main(argv=None):
    p = argparse.ArgumentParser("edideck")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("studio", help="open the Studio: any video file or link, review, render")
    s.add_argument("--port", type=int, default=8765)
    a = sub.add_parser("analyze", help="find highlights in a video file or link")
    a.add_argument("video", help="path or URL (YouTube etc.)")
    a.add_argument("-n", type=int, default=5)
    a.add_argument("-o", "--out", default="edideck-out")
    a.add_argument("--model", default="small")
    r = sub.add_parser("render", help="render a clips.json")
    r.add_argument("clips_json")
    r.add_argument("--crf", type=int, default=20)
    r.add_argument("--style", choices=list(core.STYLES), default="bold")
    r.add_argument("--watermark", default="")
    r.add_argument("--bgm", help="background music file")
    sub.add_parser("doctor", help="check what is installed")
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        for name, ok in [("ffmpeg", shutil.which("ffmpeg")), ("faster-whisper (speech)", importlib.util.find_spec("faster_whisper")),
                         ("opencv (face tracking)", importlib.util.find_spec("cv2")), ("yt-dlp (links)", importlib.util.find_spec("yt_dlp"))]:
            print(f"{'ok     ' if ok else 'missing'}  {name}")
        return
    if args.cmd == "studio":
        from .server import serve
        return serve(args.port)
    if args.cmd == "analyze":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        src = core.download(args.video, out) if args.video.startswith("http") else Path(args.video)
        base = os.environ.get("EDIDECK_LLM_URL")
        llm = (base, os.environ.get("EDIDECK_LLM_MODEL", "llama3.2")) if base else None
        clips = core.choose(core.transcribe(src, args.model), args.n, llm)
        for c in clips:
            c.track = core.track(src, c.start, c.end)
        (out / "clips.json").write_text(json.dumps({"source": str(src.resolve()), "clips": [asdict(c) for c in clips]}, indent=2), encoding="utf-8")
        print(f"Found {len(clips)} clips in {out / 'clips.json'}. Review them with: edideck studio")
        return
    path = Path(args.clips_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    for c in data["clips"]:
        print("rendered", core.render(data["source"], c, path.parent / "clips", args.crf, args.style, args.watermark, args.bgm))
