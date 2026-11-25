# Edideck

Turn long videos into short vertical clips and **review every cut, watching the real footage, before it renders.**
Local-first, free, no API keys. Original MIT code.

```bash
pip install -e ".[asr,vision,web]"     # also needs ffmpeg on PATH
edideck doctor
edideck studio                       # drop ANY video (mp4, mov, mkv...) or paste a YouTube link
```
CLI only: `edideck analyze <file-or-link> -n 5`, then `edideck render edideck-out/clips.json --style pop --watermark @me --bgm music.mp3`.

**Features:** link or file input (yt-dlp, ffmpeg) - word-level speech-to-text (Whisper) - rule-based highlight scoring,
optionally re-ranked by any local OpenAI-compatible model (`EDIDECK_LLM_URL=http://localhost:11434/v1`, Ollama works) -
face tracking that glides the 9:16 crop - 3 karaoke caption styles - text watermark - looped background music - thumbnails.

**Not yet:** multi-speaker layouts, B-roll, AI voice-over. Clip only content you own or may reuse; follow each platform's terms.
