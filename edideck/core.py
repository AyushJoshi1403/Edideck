"""Edideck core: transcribe -> pick -> caption -> frame -> render."""
from __future__ import annotations
import json, re, statistics, subprocess, sys, urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HOOKS = {"secret", "mistake", "never", "always", "truth", "why", "how", "nobody",
         "biggest", "stop", "real", "wrong", "best", "worst", "free", "need"}


@dataclass
class Word:
    w: str
    s: float
    e: float


@dataclass
class Clip:
    id: int
    start: float
    end: float
    score: float
    title: str
    words: list = field(default_factory=list)
    track: list = field(default_factory=list)  # [[seconds, face position 0..1], ...]


def transcribe(path, model="small", device="auto"):
    """Speech to word timestamps. ffmpeg decodes the audio itself, so any video format works."""
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit("Speech support missing: pip install 'edideck[asr]'")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode:
        raise RuntimeError("ffmpeg could not read the audio: " + r.stderr.decode(errors="ignore")[-300:])
    audio = np.frombuffer(r.stdout, dtype=np.float32)
    segs, _ = WhisperModel(model, device=device).transcribe(audio, word_timestamps=True, vad_filter=True)
    return [Word(x.word.strip(), x.start, x.end) for s in segs for x in (s.words or [])]


def sentences(words):
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1].s - w.e if i + 1 < len(words) else 9
        if re.search(r"[.?!]$", w.w) or gap > 0.8:
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def score(ws):
    dur = ws[-1].e - ws[0].s
    if dur <= 0:
        return 0.0
    toks = [re.sub(r"\W", "", w.w.lower()) for w in ws]
    pause = sum(max(0, b.s - a.e) for a, b in zip(ws, ws[1:])) / dur
    value = (min(len(ws) / dur, 4) * 10 + sum(t in HOOKS for t in toks) * 3
             + sum("?" in w.w for w in ws) * 4 + sum(any(c.isdigit() for c in w.w) for w in ws) * 2
             + sum(t in HOOKS for t in toks[:12]) * 2 - pause * 40)
    return round(value, 2)


def pick(words, n=5, lo=20, hi=55):
    sents, cands = sentences(words), []
    for i in range(len(sents)):
        ws = []
        for j in range(i, len(sents)):
            ws = ws + sents[j]
            d = ws[-1].e - ws[0].s
            if d > hi:
                break
            if d >= lo:
                cands.append((score(ws), ws))
    cands.sort(key=lambda c: -c[0])
    chosen = []
    for sc, ws in cands:
        if all(ws[-1].e <= c.start or ws[0].s >= c.end for c in chosen):
            chosen.append(Clip(len(chosen) + 1, ws[0].s, ws[-1].e, sc,
                               " ".join(w.w for w in ws[:7]), [w.__dict__ for w in ws]))
        if len(chosen) == n:
            break
    return chosen


def download(url, dest):
    """Fetch a video from a link (YouTube and many other sites) with yt-dlp."""
    dest = Path(dest)
    for old in dest.glob("source.*"):
        old.unlink()
    r = subprocess.run([sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format",
                        "mp4", "--no-playlist", "-o", str(dest / "source.%(ext)s"), url], capture_output=True, text=True)
    files = sorted(dest.glob("source.*"))
    if r.returncode or not files:
        raise RuntimeError("Download failed. Is yt-dlp installed (pip install 'edideck[web]')? " + r.stderr[-300:])
    return files[0]


def proxy(video, dest):
    """Small browser-friendly MP4 so any input format (MOV, MKV, ...) can be previewed."""
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vf", "scale=-2:720", "-c:v", "libx264", "-preset",
                    "veryfast", "-crf", "28", "-c:a", "aac", "-movflags", "+faststart", str(dest)],
                   capture_output=True, check=True)


def rerank(clips, base_url, model, key="none"):
    """Ask any OpenAI-compatible model (e.g. a local Ollama) to rate each clip 1-10."""
    for c in clips:
        text = " ".join(w["w"] for w in c.words)[:1500]
        body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content":
                "Rate 1-10 how strong this is as a standalone short video (hook, clarity, payoff). "
                "Reply with only the number.\n\n" + text}]}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body,
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            m = re.search(r"\d+(\.\d+)?", json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"])
            if m:
                c.score = round(float(m.group()) * 10 + c.score * 0.3, 2)
        except Exception:
            pass  # model unreachable: keep the rule-based score
    return sorted(clips, key=lambda c: -c.score)


def choose(words, n=5, llm=None):
    pool = pick(words, n * 3 if llm else n)
    if llm:
        pool = rerank(pool, *llm)[:n]
    for i, c in enumerate(pool, 1):
        c.id = i
    return pool


STYLES = {"bold": (78, "33C9FF", "FFFFFF", 420, False), "clean": (60, "FFB347", "FFFFFF", 300, False),
          "pop": (92, "FF3CAC", "FFFFFF", 520, True)}  # size, highlight (BGR), base (BGR), margin, uppercase


def _head(styles):
    fmt = ("Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,"
           "StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding")
    return ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: " + fmt +
            "\n" + styles + "\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")


def _style(name, size, hi, base, mv, align=2):
    return f"Style: {name},Arial,{size},&H00{hi},&H00{base},&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,6,0,{align},60,60,{mv},1\n"


def ass(words, start, end, style="bold", watermark="", group=4):
    """Karaoke captions: the spoken word fills with the highlight colour as it is said."""
    size, hi, base, mv, upper = STYLES.get(style, STYLES["bold"])

    def t(x):
        x = max(0.0, x - start)
        return f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"
    ws = [x for x in words if x.s >= start - 0.05 and x.e <= end + 0.05]
    rows = []
    for i in range(0, len(ws), group):
        c = ws[i:i + group]
        text = "".join(f"{{\\k{max(1, round((x.e - x.s) * 100))}}}{x.w.upper() if upper else x.w} " for x in c).strip()
        rows.append(f"Dialogue: 0,{t(c[0].s)},{t(c[-1].e)},K,,0,0,0,,{text}")
    if watermark:
        rows.append(f"Dialogue: 1,0:00:00.00,9:59:00.00,W,,0,0,0,,{{\\an9\\pos(1020,70)}}{watermark}")
    return _head(_style("K", size, hi, base, mv) + _style("W", 44, "FFFFFF", "FFFFFF", 0, 9)) + "\n".join(rows) + "\n"


def track(video, start, end, step=1.5):
    """Smoothed face position over time: [[seconds from start, 0..1], ...]; [] without OpenCV/faces."""
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(str(video))
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    pts, t, last = [], start, None
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        found = det.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.2, 5,
                                     minSize=(int(H * 0.1), int(H * 0.1)))
        if len(found):
            x, _, w, _ = max(found, key=lambda b: b[2] * b[3])
            f = (x + w / 2) / W
            last = f if last is None else last * 0.6 + f * 0.4  # smoothing: no jitter
        if last is not None:
            pts.append([round(t - start, 2), round(last, 3)])
        t += step
    cap.release()
    return pts


def _crop_x(pts):
    """ffmpeg expression that glides the crop window along the face track."""
    if not pts:
        return "(iw-ow)/2"
    expr = f"{pts[-1][1]}*iw-ow/2"
    for (t0, f0), (t1, f1) in reversed(list(zip(pts, pts[1:]))):
        expr = f"if(lt(t\\,{t1})\\,({f0}+({f1}-{f0})*(t-{t0})/{t1 - t0})*iw-ow/2\\,{expr})"
    return f"max(0\\,min(iw-ow\\,{expr}))"


def render(video, clip, out_dir, crf=20, style="bold", watermark="", bgm=None, thumb=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = clip["id"]
    (out / f"clip{n}.ass").write_text(ass([Word(**x) for x in clip["words"]], clip["start"], clip["end"],
                                          style, watermark), encoding="utf-8")
    vf = f"crop=ih*9/16:ih:{_crop_x(clip.get('track') or [])}:0,scale=1080:1920,ass=clip{n}.ass"
    cmd = ["ffmpeg", "-y", "-ss", f"{clip['start']:.3f}", "-i", str(Path(video).resolve())]
    if bgm:
        cmd += ["-stream_loop", "-1", "-i", str(Path(bgm).resolve())]
    cmd += ["-t", f"{clip['end'] - clip['start']:.3f}"]
    if bgm:
        cmd += ["-filter_complex", f"[0:v]{vf}[v];[0:a]volume=1[a0];[1:a]volume=0.12[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[a]", "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", f"clip{n}.mp4"]
    r = subprocess.run(cmd, cwd=out, capture_output=True, text=True)
    if r.returncode:
        (out / f"clip{n}.log").write_text(r.stderr, encoding="utf-8")
        raise RuntimeError(f"ffmpeg failed for clip {n}; see {out}/clip{n}.log")
    if thumb:
        (out / f"thumb{n}.ass").write_text(_head(_style("T", 96, "33C9FF", "FFFFFF", 700)) +
                                           f"Dialogue: 0,0:00:00.00,0:00:10.00,T,,0,0,0,,{clip['title']}\n", encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-i", f"clip{n}.mp4", "-ss", "1", "-frames:v", "1", "-vf", f"ass=thumb{n}.ass",
                        f"thumb{n}.jpg"], cwd=out, capture_output=True)
    return out / f"clip{n}.mp4"
