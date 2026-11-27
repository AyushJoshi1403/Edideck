"""Local Studio server (stdlib only). Listens on 127.0.0.1, so only this computer can reach it."""
import json, os, threading, uuid, webbrowser
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
from . import core

WORK = Path("edideck-work")
JOBS = {}
PAGE = Path(__file__).resolve().parent.parent / "studio" / "index.html"


def step(job, msg, pct):
    """Report progress; stop on Cancel; wait here while paused."""
    if job.get("cancel"):
        raise core.Cancelled()
    if job.get("pause"):
        job["paused"] = True
        while job.get("pause") and not job.get("cancel"):
            time.sleep(0.3)
        job["paused"] = False
        if job.get("cancel"):
            raise core.Cancelled()
    job.update(msg=msg, pct=int(pct))


# def step(job, msg, pct):
#     """Report progress, and stop here if the user pressed Cancel."""
#     if job.get("cancel"):
#         raise core.Cancelled()
#     job.update(msg=msg, pct=int(pct))


def start(fn):
    jid = uuid.uuid4().hex[:8]
    job = JOBS[jid] = {"status": "running", "msg": "Starting", "pct": 0}

    def go():
        try:
            job.update(data=fn(job), status="done", pct=100)
        except core.Cancelled:
            job.update(status="cancelled", msg="Cancelled")
        except BaseException as e:  # includes SystemExit from missing extras
            job.update(status="error", msg=str(e) or type(e).__name__)
    threading.Thread(target=go, daemon=True).start()
    return jid


def analyze(job, src, url, n, model):
    if url:
        step(job, "Downloading the video", 3)
        src = core.download(url, WORK)
    step(job, "Preparing a preview", 15)
    core.proxy(src, WORK / "proxy.mp4")
    step(job, "Listening to the audio (the first run downloads the speech model)", 25)
    words = core.transcribe(src, model, progress=lambda f: step(job, "Listening to the audio", 25 + f * 60))
    step(job, "Choosing highlights", 88)
    base = os.environ.get("EDIDECK_LLM_URL")
    llm = (base, os.environ.get("EDIDECK_LLM_MODEL", "llama3.2")) if base else None
    clips = core.choose(words, n, llm)
    for i, c in enumerate(clips):
        step(job, f"Tracking faces ({i + 1} of {len(clips)})", 90 + 9 * i / max(1, len(clips)))
        c.track = core.track(src, c.start, c.end)
    return {"source": str(Path(src).resolve()), "clips": [asdict(c) for c in clips]}


def render(job, data, opts):
    done, total = [], len(data["clips"])
    for i, c in enumerate(data["clips"]):
        step(job, f"Rendering clip {i + 1} of {total}", 100 * i / total)
        p = core.render(data["source"], c, WORK / "clips", style=opts.get("style", "bold"),
                        watermark=opts.get("watermark", ""), bgm=opts.get("bgm") or None)
        done.append({"id": c["id"], "video": "/files/" + p.name, "thumb": f"/files/thumb{c['id']}.jpg"})
    return done


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path, ctype):
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        size = path.stat().st_size
        a, b, code = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            s, _, e = rng[6:].partition("-")
            a, b, code = int(s or 0), int(e) if e else size - 1, 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = f.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser cancelled a range request

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._file(PAGE, "text/html; charset=utf-8")
        if self.path == "/api/video":
            return self._file(WORK / "proxy.mp4", "video/mp4")
        if self.path == "/api/project":  # last saved review, if its preview video still exists
            f = WORK / "project.json"
            ok = f.is_file() and (WORK / "proxy.mp4").is_file()
            return self._json({"data": json.loads(f.read_text(encoding="utf-8")) if ok else None})
        if self.path.startswith("/api/job/"):
            return self._json(JOBS.get(self.path.split("/")[-1], {"status": "error", "msg": "unknown job"}))
        if self.path.startswith("/files/"):
            name = Path(self.path[7:]).name
            return self._file(WORK / "clips" / name, "image/jpeg" if name.endswith(".jpg") else "video/mp4")
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if self.path.startswith("/api/cancel/"):
            job = JOBS.get(self.path.split("/")[-1])
            if job:
                job["cancel"] = True
            return self._json({"ok": bool(job)})
        if self.path.startswith(("/api/pause/", "/api/resume/")):
            job = JOBS.get(self.path.split("/")[-1])
            if job:
                job["pause"] = self.path.startswith("/api/pause/")
            return self._json({"ok": bool(job)})
        if self.path == "/api/project":
            WORK.mkdir(exist_ok=True)
            tmp = WORK / "project.tmp"
            tmp.write_bytes(self.rfile.read(length))
            tmp.replace(WORK / "project.json")
            return self._json({"ok": True})
        if self.path == "/api/render":
            body = json.loads(self.rfile.read(length))
            return self._json({"job": start(lambda j: render(j, body["data"], body.get("opts", {})))})
        if self.path == "/api/analyze":
            WORK.mkdir(exist_ok=True)
            if "json" in self.headers.get("Content-Type", ""):
                body = json.loads(self.rfile.read(length))
                args = (None, body["url"], int(body.get("n", 5)), body.get("model", "small"))
            else:
                for old in WORK.glob("source.*"):
                    old.unlink()
                src = WORK / ("source" + Path(unquote(self.headers.get("X-Name", "v.mp4"))).suffix[:8])
                with open(src, "wb") as f:
                    left = length
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                args = (src, None, int(self.headers.get("X-N", 5)), self.headers.get("X-Model", "small"))
            return self._json({"job": start(lambda j: analyze(j, *args))})
        self._json({"error": "not found"}, 404)


def serve(port=8765):
    url = f"http://127.0.0.1:{port}"
    print("Edideck Studio running at", url, "(Ctrl+C to stop)")
    webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
