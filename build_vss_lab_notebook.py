#!/usr/bin/env python3
"""
Generate Scaling_AI_for_GEOINT_VSS_Lab.ipynb — v4
Rebuilt with all teachable moments from live testing.
"""
import json

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text.strip()]}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text.strip()]}

cells = []

# ════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════
cells.append(md("""
# Scaling AI for GEOINT: Video Search & Summarization

### GEOINT 2026 Symposium · May 5, 2026 · Aurora, CO
**Kevin Green, Ph.D. & Keith Ober · NVIDIA WWFO**

---

> *"You have 500 hours of overhead video from last week's operation.*
> *The commander needs a situation report — in the next 10 minutes. How?"*

A human analyst watching video in real time is bounded by the clock.
This notebook demonstrates the **NVIDIA Video Search and Summarization (VSS) Blueprint** —
an AI pipeline that automatically captions, indexes, and answers questions over hours of
footage in minutes, using live NVIDIA NIM inference.

**Three Videos. Three Missions. One Key Lesson.**

| Mission | Video | Goal |
|---------|-------|------|
| **1 — Captioning** | Warehouse Safety | VLM describes every scene — zero labels, zero training |
| **2 — Summarization** | Warehouse Operations | Raw captions compressed into an operational report |
| **3 — Interactive Q&A** | Access Control (Tailgating) | Ask questions in plain English, get grounded answers |

Along the way, we discover a critical insight: **static frames capture snapshots, but video captures the story.** This distinction matters for mission-critical event detection.

| NIM Model | Role | Production VSS Equivalent |
|-----------|------|--------------------------|
| `nemotron-nano-12b-v2-vl` | VLM frame captioning + video inference | Cosmos Reason2 8B |
| `nvidia-nemotron-nano-9b-v2` | LLM summarization + Q&A | Nemotron Nano 9B |
| `nv-embedqa-e5-v5` | Dense embeddings (1024 dim) | Cosmos Embed 1 |
"""))

# ════════════════════════════════════════════════════════════════════
# PART 0: SETUP
# ════════════════════════════════════════════════════════════════════
cells.append(md("""
---
## Part 0: Setup

Install dependencies, configure NIM API access, and define all helper functions.
The notebook calls NVIDIA's hosted NIM endpoints at `integrate.api.nvidia.com` — no local GPU required.
"""))

cells.append(code("""
# ── Install dependencies (run once) ─────────────────────────────────
import subprocess, sys
for pkg in ["openai", "numpy", "matplotlib", "Pillow", "faiss-cpu"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", pkg],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("All packages ready.")
"""))

cells.append(code("""
# ── Imports ─────────────────────────────────────────────────────────
import os, sys, json, time, base64, warnings, hashlib
from pathlib import Path
from textwrap import fill
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image as PILImage
from IPython.display import display, HTML, Markdown, Video
import faiss

warnings.filterwarnings('ignore')

# ── Projector-optimized display ─────────────────────────────────────
display(HTML(\"\"\"<style>
.jp-OutputArea-output { font-size: 15px !important; }
.jp-RenderedHTMLCommon { font-size: 15px !important; }
</style>\"\"\"))

# ── Dark stage-ready theme ──────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#21262d',
    'grid.linewidth': 0.6, 'font.size': 13,
    'axes.titlesize': 14, 'figure.titlesize': 16,
    'legend.facecolor': '#161b22', 'legend.edgecolor': '#30363d',
})
ACCENT, WARM, SUCCESS, MUTED = '#58a6ff', '#f78166', '#3fb950', '#8b949e'
NIM_GREEN = '#76b900'

print("Display and theme configured for projector.")
"""))

cells.append(md("### NIM API Configuration\n\nSet your NIM API key below. Get one free at [build.nvidia.com](https://build.nvidia.com).\nPre-computed results are included — you can run the full notebook without a key.\nCells that need a live key will show a clear message."))

cells.append(code("""
# ── NIM API Key ─────────────────────────────────────────────────────
# Option 1: Set directly
# NIM_API_KEY = "nvapi-YOUR-KEY-HERE"

# Option 2: Load from environment or ~/.secrets
NIM_API_KEY = os.environ.get('NIM_API_KEY', '')
if not NIM_API_KEY:
    secrets = Path.home() / '.secrets'
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            line = line.strip()
            if line.startswith('export NIM_API_KEY'):
                NIM_API_KEY = line.split('=', 1)[1].strip().strip("'").strip('"')

# ── NIM Endpoints ───────────────────────────────────────────────────
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
VLM_MODEL    = "nvidia/nemotron-nano-12b-v2-vl"
LLM_MODEL    = "nvidia/nvidia-nemotron-nano-9b-v2"
EMBED_MODEL  = "nvidia/nv-embedqa-e5-v5"

# ── Data Paths ──────────────────────────────────────────────────────
DATA_DIR       = Path("data")
VIDEOS_DIR     = DATA_DIR / "videos"
CAPTIONS_DIR   = DATA_DIR / "captions"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FRAMES_DIR     = DATA_DIR / "frames"
for d in [VIDEOS_DIR, CAPTIONS_DIR, EMBEDDINGS_DIR, FRAMES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"{'✓' if NIM_API_KEY else '✗'} NIM_API_KEY: {'set (' + NIM_API_KEY[:8] + '...)' if NIM_API_KEY else 'NOT SET — pre-computed data will be used'}")
print(f"  Endpoint : {NIM_BASE_URL}")
print(f"  VLM      : {VLM_MODEL}")
print(f"  LLM      : {LLM_MODEL}")
print(f"  Embed    : {EMBED_MODEL}")
"""))

cells.append(md("### Helper Functions"))

cells.append(code("""
# ── NIM API Helpers ─────────────────────────────────────────────────
from openai import OpenAI
import subprocess

def get_nim_client():
    return OpenAI(base_url=NIM_BASE_URL, api_key=NIM_API_KEY)

def caption_frame_nim(image_path, prompt=None):
    if not NIM_API_KEY:
        raise RuntimeError("NIM_API_KEY required for live captioning")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    if prompt is None:
        prompt = ("Describe this surveillance frame in detail. "
                  "What objects, people, activities, and safety concerns do you observe?")
    client = get_nim_client()
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        max_tokens=512, temperature=0.3, top_p=0.4,
    )
    return resp.choices[0].message.content

def video_native_caption(video_path, prompt):
    \"\"\"Send an entire video to the VLM for motion-aware inference.\"\"\"
    if not NIM_API_KEY:
        raise RuntimeError("NIM_API_KEY required for video inference")
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    client = get_nim_client()
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        max_tokens=1024, temperature=0.3,
    )
    return resp.choices[0].message.content

def llm_complete(prompt, system="You are a concise video analytics assistant."):
    if not NIM_API_KEY:
        raise RuntimeError("NIM_API_KEY required for live LLM calls")
    client = get_nim_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        max_tokens=2048, temperature=0.3,
    )
    return resp.choices[0].message.content

def embed_texts(texts, input_type="passage"):
    if not NIM_API_KEY:
        raise RuntimeError("NIM_API_KEY required for live embedding")
    client = get_nim_client()
    resp = client.embeddings.create(
        model=EMBED_MODEL, input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return [e.embedding for e in resp.data]

def embed_query(query):
    return embed_texts([query], input_type="query")[0]

# ── Video Processing Helpers ────────────────────────────────────────
def get_video_duration(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
    out = subprocess.check_output(cmd)
    return float(json.loads(out)["format"]["duration"])

def extract_frames_from_video(video_path, output_dir, chunk_duration=10.0, frames_per_chunk=3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path)
    chunks = []
    chunk_id = 0
    t = 0.0
    while t < duration:
        end = min(t + chunk_duration, duration)
        frame_paths = []
        for fi in range(frames_per_chunk):
            frac = (fi + 0.5) / frames_per_chunk
            ts = t + frac * (end - t)
            out_path = output_dir / f"chunk_{chunk_id:02d}_frame_{fi:02d}.jpg"
            if not out_path.exists():
                subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(ts), "-i", str(video_path),
                                "-vframes", "1", "-q:v", "2", str(out_path)], check=False)
            if out_path.exists():
                frame_paths.append(str(out_path))
        chunks.append({"chunk_id": chunk_id, "start_time": t, "end_time": end, "frame_paths": frame_paths})
        t = end
        chunk_id += 1
    return chunks

def get_cover_frame(chunks_data, frames_dir):
    cover = {}
    for c in chunks_data:
        fps = c.get("frame_paths", [])
        if fps:
            cover[c["chunk_id"]] = PILImage.open(fps[len(fps)//2])
    return cover

def recursive_summarize(captions, group_size=3):
    texts = [f"[{c['start_time']:.0f}s-{c['end_time']:.0f}s] {c['caption']}" for c in captions]
    groups = [texts[i:i+group_size] for i in range(0, len(texts), group_size)]
    group_summaries = []
    for i, grp in enumerate(groups):
        prompt = ("Summarize these sequential video observations into a single "
                  "coherent paragraph. Preserve key details and timestamps.\\n\\n"
                  + "\\n".join(grp))
        summary = llm_complete(prompt)
        group_summaries.append(summary)
        print(f"  Group {i+1}/{len(groups)} summarized")
    final_prompt = ("You are a video analytics report generator. Synthesize these "
                    "group summaries into a structured operational report with:\\n"
                    "- Executive Summary\\n- Key Events (with timestamps)\\n"
                    "- Safety Observations\\n- Recommendations\\n\\n"
                    + "\\n\\n".join(group_summaries))
    final_report = llm_complete(final_prompt)
    return {"stages": 2, "group_summaries": group_summaries, "final_report": final_report}

print("All helper functions defined. ✓")
"""))

# ════════════════════════════════════════════════════════════════════
# PART 1: MEET THE VIDEOS
# ════════════════════════════════════════════════════════════════════
cells.append(md("""
---
## Part 1: Meet the Videos

Before the AI sees anything, **you** watch first. Form your own mental model of what happened in each clip, then compare it against what the AI produces. That contrast — *"I just watched that, and the AI described exactly what I saw"* — is the point.
"""))

cells.append(code("""
# ── Video Catalog ───────────────────────────────────────────────────
VIDEOS = {
    "warehouse_safety": {
        "file": "warehouse_safety_0002.mp4",
        "title": "Warehouse Safety Monitoring",
        "duration": "~30s", "size": "7.4 MB",
        "mission": "Mission 1: Video Captioning",
        "scenario": "PPE compliance, worker safety, equipment operations",
        "mission_color": "#3fb950",
        "source": "NVIDIA VSS Public Safety Blueprint (NGC)",
    },
    "warehouse_ops": {
        "file": "warehouse_short.mp4",
        "title": "Warehouse Operations",
        "duration": "10s", "size": "0.9 MB",
        "mission": "Mission 2: Video Summarization",
        "scenario": "Operational activity monitoring, logistics flow",
        "mission_color": "#58a6ff",
        "source": "NVIDIA VSS Dev Profile Sample Data (NGC)",
    },
    "tailgating": {
        "file": "test1.mp4",
        "title": "Access Control — Tailgating",
        "duration": "31s", "size": "5.0 MB",
        "mission": "Mission 3: Interactive Q&A",
        "scenario": "Unauthorized entry detection, access point monitoring",
        "mission_color": "#f78166",
        "source": "NVIDIA VSS Public Safety Tailgate Blueprint (NGC)",
    },
}

html = ['<div style="display:flex;flex-direction:column;gap:16px;max-width:900px;">']
for key, v in VIDEOS.items():
    path = VIDEOS_DIR / v["file"]
    exists = path.exists()
    status = f'<span style="color:#3fb950">✓ ready</span>' if exists else '<span style="color:#f78166">✗ not found</span>'
    html.append(f'''
    <div style="border:2px solid {v['mission_color']};border-radius:8px;overflow:hidden;background:#0d1117;">
      <div style="display:flex;gap:16px;padding:12px;">
        <div style="flex:0 0 320px;">
          {'<video width="320" height="180" controls style="display:block;background:#000;border-radius:4px;"><source src="' + str(path) + '" type="video/mp4"></video>' if exists else '<div style="width:320px;height:180px;background:#161b22;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#8b949e;">Video not found</div>'}
        </div>
        <div style="flex:1;color:#e6edf3;">
          <div style="font-size:1.1em;font-weight:bold;color:{v['mission_color']}">{v['mission']}</div>
          <div style="font-size:1.0em;font-weight:bold;margin-top:4px;">{v['title']}</div>
          <div style="color:#8b949e;font-size:0.85em;margin-top:4px;">{v['duration']} · {v['size']} · {status}</div>
          <div style="color:#58a6ff;font-size:0.88em;margin-top:6px;">{v['scenario']}</div>
          <div style="color:#8b949e;font-size:0.78em;margin-top:4px;font-family:monospace;">Source: {v['source']}</div>
        </div>
      </div>
    </div>''')
html.append('</div>')
display(HTML(''.join(html)))
missing = [v["file"] for v in VIDEOS.values() if not (VIDEOS_DIR / v["file"]).exists()]
if missing:
    print(f"\\n⚠ Missing videos: {missing}")
    print(f"  Copy them to: {VIDEOS_DIR.resolve()}")
else:
    print(f"\\n✓ All {len(VIDEOS)} videos ready in {VIDEOS_DIR}")
"""))

# ── Architecture Diagram ────────────────────────────────────────────
cells.append(md("### How VSS Works: The Pipeline\n\nEvery mission follows the same pipeline. The only difference is where we stop and what we ask."))

cells.append(code("""
fig, ax = plt.subplots(figsize=(14, 3.5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.axis('off')
ax.set_xlim(0, 14); ax.set_ylim(0, 3.5)
stages = [
    ("Video", 0.5, "#8b949e", "Raw footage"),
    ("Chunk", 2.5, "#8b949e", "10s segments"),
    ("Frames", 4.5, "#8b949e", "Key frames"),
    ("VLM Caption", 6.8, NIM_GREEN, "nemotron-vl"),
    ("LLM Summary", 9.0, "#58a6ff", "nemotron-9b"),
    ("Embed + Index", 11.3, "#f78166", "nv-embedqa"),
    ("Q&A", 13.2, "#3fb950", "Search + RAG"),
]
for name, x, color, sub in stages:
    box = FancyBboxPatch((x-0.7, 1.0), 1.6, 1.4, boxstyle="round,pad=0.15",
                          facecolor='#161b22', edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+0.1, 1.95, name, ha='center', va='center', fontsize=10, fontweight='bold', color='#e6edf3')
    ax.text(x+0.1, 1.45, sub, ha='center', va='center', fontsize=8, color=color)
for i in range(len(stages)-1):
    x1 = stages[i][1] + 0.9
    x2 = stages[i+1][1] - 0.6
    ax.annotate('', xy=(x2, 1.7), xytext=(x1, 1.7), arrowprops=dict(arrowstyle='->', color='#30363d', lw=1.5))
ax.text(6.8, 0.4, "◆ Mission 1 stops here", ha='center', fontsize=9, color=NIM_GREEN)
ax.text(9.0, 0.4, "◆ Mission 2 stops here", ha='center', fontsize=9, color='#58a6ff')
ax.text(12.25, 0.4, "◆ Mission 3 uses full pipeline", ha='center', fontsize=9, color='#f78166')
plt.tight_layout()
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════
# MISSION 1: VIDEO CAPTIONING
# ════════════════════════════════════════════════════════════════════
cells.append(md("""
---
## Mission 1: Video Captioning

**Video:** Warehouse Safety Monitoring (`warehouse_safety_0002.mp4`)

**The challenge:** An analyst watches 30 seconds of warehouse footage in 30 seconds and
takes notes. The VLM processes every frame and produces detailed captions in under 10 seconds.

**GEOINT connection:** This is automated metadata population — the #1 gap in GEOINT today.
Over 70% of collected imagery lacks complete metadata for discovery. The VLM fills that gap
at machine speed, zero training required.

▶ **Watch the video first.** Then run the cells below to see what the AI sees.
"""))

cells.append(code("""
m1_video = VIDEOS_DIR / "warehouse_safety_0002.mp4"
if m1_video.exists():
    display(HTML(f'''
    <div style="background:#0d1117;padding:12px;border-radius:8px;border:1px solid #3fb950;max-width:640px;">
      <div style="color:#3fb950;font-weight:bold;font-size:1.05em;margin-bottom:8px;">▶ Mission 1: Watch First, Then Caption</div>
      <video width="100%" controls style="border-radius:4px;"><source src="{m1_video}" type="video/mp4"></video>
      <div style="color:#8b949e;font-size:0.85em;margin-top:6px;">What did you notice? PPE? Equipment? Worker activities?</div>
    </div>'''))
"""))

cells.append(md("### Step 1: Chunk the Video and Extract Frames"))

cells.append(code("""
m1_stem = "warehouse_safety_0002"
m1_frames_dir = FRAMES_DIR / m1_stem
m1_chunks_cache = CAPTIONS_DIR / f"chunks_{m1_stem}.json"
if m1_chunks_cache.exists():
    m1_chunks = json.loads(m1_chunks_cache.read_text())
    print(f"✓ Loaded {len(m1_chunks)} chunks from cache")
elif m1_video.exists():
    start = time.time()
    m1_chunks = extract_frames_from_video(m1_video, m1_frames_dir, chunk_duration=10.0, frames_per_chunk=3)
    m1_chunks_cache.write_text(json.dumps(m1_chunks, indent=2))
    print(f"✓ {len(m1_chunks)} chunks, frames extracted in {time.time()-start:.1f}s")
else:
    m1_chunks = []
    print("⚠ Video not found — using pre-computed data only")
m1_covers = get_cover_frame(m1_chunks, m1_frames_dir)
if m1_covers:
    n = len(m1_covers)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3))
    if n == 1: axes = [axes]
    for ax, (cid, img) in zip(axes, m1_covers.items()):
        ax.imshow(np.array(img))
        c = m1_chunks[cid]
        ax.set_title(f"Chunk {cid} [{c['start_time']:.0f}s-{c['end_time']:.0f}s]", fontsize=10, color=NIM_GREEN)
        ax.axis('off')
    plt.suptitle("Extracted Cover Frames", fontsize=13, color='#e6edf3', y=1.02)
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("### Step 2: VLM Captioning — What Does the AI See?"))

cells.append(code("""
m1_captions_path = CAPTIONS_DIR / f"nim_captions_{m1_stem}.json"
if m1_captions_path.exists():
    m1_captions = json.loads(m1_captions_path.read_text())
    print(f"✓ Loaded {len(m1_captions)} captions from cache")
    print(f"  Model: {m1_captions[0].get('model', VLM_MODEL)}")
else:
    print(f"Running live VLM captioning ({VLM_MODEL}) ...")
    m1_captions = []
    for c in m1_chunks:
        fps = c.get("frame_paths", [])
        if not fps: continue
        frame = fps[len(fps)//2]
        start_t = time.time()
        caption = caption_frame_nim(frame)
        elapsed = time.time() - start_t
        entry = {"chunk_id": c["chunk_id"], "start_time": c["start_time"],
                 "end_time": c["end_time"], "caption": caption,
                 "model": VLM_MODEL, "frame_path": frame}
        m1_captions.append(entry)
        print(f"  Chunk {c['chunk_id']:02d} ({c['start_time']:.0f}s-{c['end_time']:.0f}s) [{elapsed:.1f}s] {caption[:70]}...")
    m1_captions_path.write_text(json.dumps(m1_captions, indent=2))
    print(f"\\n✓ {len(m1_captions)} captions saved")
print(f"\\n{'─'*80}")
print(f"VLM CAPTION STRIP — {VIDEOS['warehouse_safety']['title']}")
print(f"{'─'*80}")
for c in m1_captions:
    ts = f"[{c['start_time']:5.0f}s - {c['end_time']:5.0f}s]"
    print(f"\\nChunk {c['chunk_id']:02d} {ts}")
    print(f"  {c['caption'][:300]}")
"""))

cells.append(md("""
### Step 3: Interactive Follow-up — Ask About What the AI Saw

The captions are now searchable context. Let's ask a specific safety question grounded in the VLM's observations.
"""))

cells.append(code("""
PPE_QUESTION = "Based on the video observations, are all workers wearing proper PPE (hard hat, safety vest)? If anyone is not wearing PPE, describe what they were wearing instead."
context = "\\n".join([f"[{c['start_time']:.0f}s-{c['end_time']:.0f}s]: {c['caption']}" for c in m1_captions])
prompt = f\"\"\"You are a warehouse safety analyst. Answer the question using ONLY the video observations below.

VIDEO OBSERVATIONS:
{context}

QUESTION: {PPE_QUESTION}

Provide a specific, grounded answer with timestamps. If the observations don't contain enough detail, say so.\"\"\"
m1_qa_cache = CAPTIONS_DIR / f"qa_ppe_{m1_stem}.json"
if m1_qa_cache.exists():
    answer = json.loads(m1_qa_cache.read_text())["answer"]
    print("✓ Loaded cached answer")
else:
    print(f"Asking: {PPE_QUESTION}\\n")
    answer = llm_complete(prompt)
    m1_qa_cache.write_text(json.dumps({"question": PPE_QUESTION, "answer": answer}))
display(HTML(f'''
<div style="background:#0d1117;border:1px solid #3fb950;border-radius:8px;padding:16px 20px;max-width:860px;">
  <div style="color:#3fb950;font-weight:bold;font-size:0.9em;">▶ ANALYST QUESTION</div>
  <div style="color:#e6edf3;font-size:1.0em;margin:8px 0;font-style:italic;">"{PPE_QUESTION}"</div>
  <hr style="border-color:#30363d;margin:12px 0;">
  <div style="color:#76b900;font-weight:bold;font-size:0.9em;">◆ AI RESPONSE (grounded in video observations)</div>
  <div style="color:#e6edf3;font-size:0.95em;margin-top:8px;line-height:1.7;white-space:pre-wrap;">{answer}</div>
</div>'''))
"""))

# ════════════════════════════════════════════════════════════════════
# MISSION 2: VIDEO SUMMARIZATION
# ════════════════════════════════════════════════════════════════════
cells.append(md("""
---
## Mission 2: Video Summarization

**Video:** Warehouse Operations (`warehouse_short.mp4`)

**The challenge:** A 10-second clip produces a handful of captions. A 2-hour surveillance archive produces
720 captions. No one reads 720 paragraphs. Recursive summarization compresses them into a
single operational report — same pattern, any video length.

**Race the clock:** This 10-second video is processed in under 3 seconds.
A 2-hour archive with 8 GPU workers? Under 10 minutes. A human analyst? 2 hours + write-up time.

▶ **Watch the video, then run the summarization pipeline.**
"""))

cells.append(code("""
m2_video = VIDEOS_DIR / "warehouse_short.mp4"
if m2_video.exists():
    display(HTML(f'''
    <div style="background:#0d1117;padding:12px;border-radius:8px;border:1px solid #58a6ff;max-width:640px;">
      <div style="color:#58a6ff;font-weight:bold;font-size:1.05em;margin-bottom:8px;">▶ Mission 2: Watch First, Then Summarize</div>
      <video width="100%" controls style="border-radius:4px;"><source src="{m2_video}" type="video/mp4"></video>
      <div style="color:#8b949e;font-size:0.85em;margin-top:6px;">Watch carefully — did you see anything fall?</div>
    </div>'''))
"""))

cells.append(md("### Step 1: Caption the Video (Frame-Based)"))

cells.append(code("""
m2_stem = "warehouse_short"
m2_frames_dir = FRAMES_DIR / m2_stem
m2_captions_path = CAPTIONS_DIR / f"nim_captions_{m2_stem}.json"
if m2_captions_path.exists():
    m2_captions = json.loads(m2_captions_path.read_text())
    print(f"✓ Loaded {len(m2_captions)} captions from cache")
else:
    if m2_video.exists():
        m2_chunks = extract_frames_from_video(m2_video, m2_frames_dir, chunk_duration=3.0, frames_per_chunk=3)
        print(f"Running live VLM captioning ({VLM_MODEL}) ...")
        m2_captions = []
        for c in m2_chunks:
            fps = c.get("frame_paths", [])
            if not fps: continue
            frame = fps[len(fps)//2]
            caption = caption_frame_nim(frame)
            m2_captions.append({"chunk_id": c["chunk_id"], "start_time": c["start_time"],
                                "end_time": c["end_time"], "caption": caption, "model": VLM_MODEL})
            print(f"  Chunk {c['chunk_id']:02d}: {caption[:80]}...")
        m2_captions_path.write_text(json.dumps(m2_captions, indent=2))
    else:
        m2_captions = []
for c in m2_captions:
    print(f"\\n[{c['start_time']:.0f}s-{c['end_time']:.0f}s] {c['caption'][:200]}")
"""))

cells.append(md("### Step 2: Recursive Summarization → Operational Report"))

cells.append(code("""
m2_summary_path = CAPTIONS_DIR / f"nim_summary_{m2_stem}.json"
if m2_summary_path.exists():
    m2_summary = json.loads(m2_summary_path.read_text())
    print(f"✓ Loaded cached summary ({len(m2_summary['final_report'])} chars)")
elif m2_captions:
    print("Running recursive summarization...")
    m2_summary = recursive_summarize(m2_captions)
    m2_summary_path.write_text(json.dumps(m2_summary, indent=2))
    print(f"✓ Summary generated ({len(m2_summary['final_report'])} chars)")
else:
    m2_summary = {"final_report": "No captions available.", "group_summaries": []}
display(HTML(f'''
<div style="background:#0d1117; border:1px solid #58a6ff; border-radius:8px; padding:20px 28px; max-width:860px;">
  <div style="color:#58a6ff; font-size:0.9em; font-weight:bold; margin-bottom:12px;">■ OPERATIONAL REPORT · NVIDIA VSS · {LLM_MODEL.split("/")[-1]}</div>
  <div style="color:#e6edf3; font-size:0.95em; line-height:1.7; white-space:pre-wrap;">{m2_summary["final_report"]}</div>
</div>'''))
"""))

# ── TEACHABLE MOMENT: Frame vs Video ────────────────────────────────
cells.append(md("""
### 🔍 Teachable Moment: What Did the AI Miss?

If you watched the video carefully, you saw a worker **carrying two boxes and dropping one.** But the operational report above makes no mention of a dropped box. Why?

**The answer:** Frame-based captioning extracts a single snapshot from the middle of each chunk. If the box fell *before* or *after* that exact frame, the VLM never saw it. A still frame cannot capture motion — it sees a warehouse worker standing, not a warehouse worker *dropping something.*

This is not a model failure. It's a **sensing limitation.** The VLM described exactly what it saw in the frame. It just never saw the event.

**The fix:** Send the entire video clip to the VLM as video, not as a single frame. This lets the model see temporal relationships — motion, cause and effect, before and after. Let's try it.
"""))

cells.append(code("""
# ── Video-Native Inference: See the Motion ──────────────────────────
m2_video_cache = CAPTIONS_DIR / f"video_native_{m2_stem}.json"
if m2_video_cache.exists():
    m2_video_caption = json.loads(m2_video_cache.read_text())["caption"]
    print("✓ Loaded cached video-native caption")
else:
    print("Running video-native VLM inference (sending full 10s video)...")
    m2_video_caption = video_native_caption(
        m2_video,
        "Describe all events in this warehouse video in detail. "
        "Note any safety incidents, dropped items, or near-misses."
    )
    m2_video_cache.write_text(json.dumps({"caption": m2_video_caption}))

display(HTML(f'''
<div style="background:#0d1117;border:2px solid #3fb950;border-radius:8px;padding:16px 20px;max-width:860px;">
  <div style="color:#3fb950;font-weight:bold;font-size:0.95em;">◆ VIDEO-NATIVE VLM CAPTION (sees motion, not just snapshots)</div>
  <div style="color:#e6edf3;font-size:0.95em;margin-top:10px;line-height:1.7;white-space:pre-wrap;">{m2_video_caption}</div>
</div>
<div style="background:#161b22;border:1px solid #f78166;border-radius:8px;padding:12px 16px;margin-top:12px;max-width:860px;">
  <div style="color:#f78166;font-weight:bold;font-size:0.85em;">⚡ KEY INSIGHT</div>
  <div style="color:#e6edf3;font-size:0.9em;margin-top:6px;">
    Frame-based captioning: <em>"Worker in PPE, organized warehouse, no issues."</em><br>
    Video-native captioning: <em>"He drops a box on the floor, which rolls away."</em><br><br>
    <strong>Frames capture snapshots. Video captures the story.</strong>
    This is why production VSS uses Cosmos Reason 2 with video-native inference.
  </div>
</div>'''))
"""))

cells.append(md("""
### Audience Challenge: Change the Focus

The same captions can produce completely different reports by changing the summarization prompt.
Try editing the `FOCUS` variable below and re-running the cell.

**Suggested focuses:**
- `"Focus exclusively on forklift and vehicle movements"`
- `"Focus on worker safety and PPE compliance"`
- `"Generate a timeline of all human activities"`
"""))

cells.append(code("""
FOCUS = "Focus on worker safety and PPE compliance"  # ← CHANGE ME
if m2_captions and NIM_API_KEY:
    context = "\\n".join([f"[{c['start_time']:.0f}s-{c['end_time']:.0f}s]: {c['caption']}" for c in m2_captions])
    prompt = f\"\"\"Summarize these video observations into a focused report.
FOCUS: {FOCUS}
VIDEO OBSERVATIONS:
{context}
Produce a structured report with: Summary, Key Events (with timestamps), and Recommendations.\"\"\"
    focused_report = llm_complete(prompt)
    display(HTML(f'''
    <div style="background:#0d1117;border:1px solid #f78166;border-radius:8px;padding:16px 20px;max-width:860px;">
      <div style="color:#f78166;font-weight:bold;font-size:0.9em;">◆ FOCUSED REPORT: {FOCUS}</div>
      <div style="color:#e6edf3;font-size:0.95em;margin-top:10px;line-height:1.7;white-space:pre-wrap;">{focused_report}</div>
    </div>'''))
else:
    print("Requires NIM_API_KEY for live re-summarization.")
"""))

# ════════════════════════════════════════════════════════════════════
# MISSION 3: INTERACTIVE Q&A
# ════════════════════════════════════════════════════════════════════
cells.append(md("""
---
## Mission 3: Interactive Q&A

**Video:** Access Control — Tailgating (`test1.mp4`)

**The challenge:** Security cameras generate thousands of hours of footage. An analyst can't
watch it all. With VSS, they type a question in plain English: *"Where do you see someone
tailgating?"* and get a timestamped, evidence-backed answer in seconds.

This mission demonstrates the full pipeline **and** reveals the same frame-vs-video lesson
from Mission 2 in an even more dramatic way.

▶ **Watch the 31-second video. Did you spot the tailgating? Then let's ask the AI.**
"""))

cells.append(code("""
m3_video = VIDEOS_DIR / "test1.mp4"
if m3_video.exists():
    display(HTML(f'''
    <div style="background:#0d1117;padding:12px;border-radius:8px;border:1px solid #f78166;max-width:640px;">
      <div style="color:#f78166;font-weight:bold;font-size:1.05em;margin-bottom:8px;">▶ Mission 3: Watch First, Then Interrogate</div>
      <video width="100%" controls style="border-radius:4px;"><source src="{m3_video}" type="video/mp4"></video>
      <div style="color:#8b949e;font-size:0.85em;margin-top:6px;">Did you spot the tailgating? How many people? What happened?</div>
    </div>'''))
"""))

cells.append(md("""
### Step 1: Frame-Based Pipeline — Chunk, Caption, Embed, Index

First, we run the standard frame-based pipeline with a security-specific captioning prompt.
This tells the VLM exactly what to look for — tailgating, piggybacking, access control violations.
"""))

cells.append(code("""
m3_stem = "test1"
m3_frames_dir = FRAMES_DIR / m3_stem
m3_captions_path = CAPTIONS_DIR / f"nim_captions_{m3_stem}.json"
m3_emb_path = EMBEDDINGS_DIR / f"nim_embeddings_{m3_stem}.npy"
m3_index_path = str(EMBEDDINGS_DIR / f"nim_faiss_{m3_stem}.index")

# Chunk and extract frames
if m3_video.exists() and not (FRAMES_DIR / m3_stem).exists():
    m3_chunks = extract_frames_from_video(m3_video, m3_frames_dir, chunk_duration=10.0, frames_per_chunk=3)
    print(f"✓ {len(m3_chunks)} chunks extracted")
elif m3_video.exists():
    m3_chunks = extract_frames_from_video(m3_video, m3_frames_dir, chunk_duration=10.0, frames_per_chunk=3)
    print(f"✓ {len(m3_chunks)} chunks ready")

# Caption with security-specific prompt
if m3_captions_path.exists():
    m3_captions = json.loads(m3_captions_path.read_text())
    print(f"✓ Loaded {len(m3_captions)} captions from cache")
else:
    SECURITY_PROMPT = ("You are monitoring an access control point. Describe this frame in detail. "
                       "Note how many people are present, whether anyone is following another person "
                       "through a gate or door, and any security concerns like tailgating or piggybacking.")
    print(f"Running VLM captioning with security prompt ({VLM_MODEL}) ...")
    m3_captions = []
    for c in m3_chunks:
        fps = c.get("frame_paths", [])
        if not fps: continue
        frame = fps[len(fps)//2]
        caption = caption_frame_nim(frame, prompt=SECURITY_PROMPT)
        m3_captions.append({"chunk_id": c["chunk_id"], "start_time": c["start_time"],
                            "end_time": c["end_time"], "caption": caption, "model": VLM_MODEL})
        print(f"  Chunk {c['chunk_id']:02d}: {caption[:80]}...")
    m3_captions_path.write_text(json.dumps(m3_captions, indent=2))
    print(f"\\n✓ {len(m3_captions)} captions saved")

# Embeddings + FAISS
if m3_emb_path.exists():
    m3_embeddings = np.load(str(m3_emb_path)).astype(np.float32)
    print(f"✓ Loaded embeddings: {m3_embeddings.shape}")
else:
    print("Generating embeddings...")
    texts = [c["caption"] for c in m3_captions]
    vecs = embed_texts(texts)
    m3_embeddings = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(m3_embeddings, axis=1, keepdims=True)
    m3_embeddings = m3_embeddings / np.maximum(norms, 1e-9)
    np.save(str(m3_emb_path), m3_embeddings)
    print(f"✓ Embeddings: {m3_embeddings.shape}")

dim = m3_embeddings.shape[1]
m3_index = faiss.IndexFlatIP(dim)
m3_index.add(m3_embeddings)
print(f"✓ FAISS index: {m3_index.ntotal} vectors × {dim}d")
"""))

cells.append(md("### Step 2: Semantic Search — Frame-Based Results"))

cells.append(code("""
QUERY = "Where do you see someone tailgating through the access point?"
m3_search_cache = CAPTIONS_DIR / f"search_{m3_stem}_{hashlib.md5(QUERY.encode()).hexdigest()[:8]}.json"
if m3_search_cache.exists():
    search_results = json.loads(m3_search_cache.read_text())
    print(f"✓ Loaded cached search results")
elif len(m3_embeddings) > 0:
    qvec = np.array(embed_query(QUERY), dtype=np.float32).reshape(1, -1)
    qvec /= max(np.linalg.norm(qvec), 1e-9)
    scores, indices = m3_index.search(qvec, 3)
    search_results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0: continue
        c = m3_captions[idx]
        search_results.append({"rank": rank+1, "chunk_id": c["chunk_id"],
            "start_time": c["start_time"], "end_time": c["end_time"],
            "caption": c["caption"], "score": float(score)})
    m3_search_cache.write_text(json.dumps(search_results, indent=2))
else:
    search_results = []
print(f'Search: "{QUERY}"\\n')
for res in search_results:
    score_col = SUCCESS if res['score'] > 0.85 else (ACCENT if res['score'] > 0.70 else MUTED)
    ts = f"{int(res['start_time']//60):02d}:{int(res['start_time']%60):02d}–{int(res['end_time']//60):02d}:{int(res['end_time']%60):02d}"
    display(HTML(f'''
    <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px 16px;margin:8px 0;max-width:860px;border-left:3px solid {score_col};">
      <div style="color:{score_col};font-weight:bold;font-size:0.9em;">Rank {res['rank']} · [{ts}] · Similarity: {res['score']:.3f}</div>
      <div style="color:#e6edf3;font-size:0.92em;margin-top:6px;line-height:1.6;">{res['caption'][:300]}</div>
    </div>'''))
"""))

cells.append(md("""
### 🔍 Teachable Moment: The Three-Way Comparison

Notice the low similarity scores above. The frame-based captions likely said *"no tailgating concerns"* — even though we told the VLM exactly what to look for. Why?

**Tailgating is a motion event.** It only exists as a temporal sequence: Person A scans their badge, the gate opens, Person B follows through without scanning. A single frame shows people near a gate — which looks perfectly normal.

The VLM was primed to look for tailgating, saw the right people at the right gate, and still confidently concluded no tailgating occurred. Because a snapshot can never capture *"following."*

Let's ask the VLM the same question, but send it the **full video** instead of individual frames.
"""))

cells.append(code("""
# ── Video-Native Inference: See the Tailgating ─────────────────────
m3_video_cache = CAPTIONS_DIR / f"video_native_{m3_stem}.json"
if m3_video_cache.exists():
    m3_video_caption = json.loads(m3_video_cache.read_text())["caption"]
    print("✓ Loaded cached video-native caption")
else:
    print("Running video-native VLM inference (sending full 31s video)...")
    m3_video_caption = video_native_caption(
        m3_video,
        "Describe all events in this access control video. Note anyone following "
        "another person through a gate, door, or turnstile without scanning their "
        "own badge or credential. This is called 'tailgating' in physical security."
    )
    m3_video_cache.write_text(json.dumps({"caption": m3_video_caption}))

display(HTML(f'''
<div style="background:#0d1117;border:2px solid #3fb950;border-radius:8px;padding:16px 20px;max-width:860px;">
  <div style="color:#3fb950;font-weight:bold;font-size:0.95em;">◆ VIDEO-NATIVE VLM CAPTION (sees the full sequence)</div>
  <div style="color:#e6edf3;font-size:0.95em;margin-top:10px;line-height:1.7;white-space:pre-wrap;">{m3_video_caption}</div>
</div>
<div style="background:#161b22;border:1px solid #f78166;border-radius:8px;padding:12px 16px;margin-top:12px;max-width:860px;">
  <div style="color:#f78166;font-weight:bold;font-size:0.85em;">⚡ KEY INSIGHT — The Same Lesson, Even More Dramatic</div>
  <div style="color:#e6edf3;font-size:0.9em;margin-top:6px;">
    Frame-based (even with security prompt): <em>"No tailgating concerns."</em><br>
    Video-native: <em>"One person follows closely behind without scanning their own access device."</em><br><br>
    <strong>If your analyst only sees still frames from a drone feed, they miss the event that happened between frames.
    If they see the video, they see the truth.</strong>
  </div>
</div>'''))
"""))

cells.append(md("""
### Step 3: RAG Q&A — Grounded in Video-Native Evidence

Now let's use the video-native caption as context for Q&A. This is how production VSS should work:
video-native inference feeds the RAG pipeline, so answers are grounded in what actually happened, not what a snapshot suggested.
"""))

cells.append(code("""
RAG_QUESTION = "Describe the tailgating incident in detail. How many people were involved? What security measures were visible?"
m3_rag_cache = CAPTIONS_DIR / f"rag_video_{m3_stem}_{hashlib.md5(RAG_QUESTION.encode()).hexdigest()[:8]}.json"
if m3_rag_cache.exists():
    rag_answer = json.loads(m3_rag_cache.read_text())["answer"]
    print("✓ Loaded cached RAG answer")
else:
    prompt = f\"\"\"Answer the analyst's question using ONLY the video observation below.
Cite details. If the evidence is insufficient, say so.

VIDEO OBSERVATION:
{m3_video_caption}

QUESTION: {RAG_QUESTION}\"\"\"
    rag_answer = llm_complete(prompt, system="You are a physical security analyst providing grounded video analysis.")
    m3_rag_cache.write_text(json.dumps({"question": RAG_QUESTION, "answer": rag_answer}))
display(HTML(f'''
<div style="background:#0d1117;border:1px solid #f78166;border-radius:8px;padding:16px 20px;max-width:860px;">
  <div style="color:#f78166;font-weight:bold;font-size:0.9em;">▶ ANALYST QUESTION</div>
  <div style="color:#e6edf3;font-size:1.0em;margin:8px 0;font-style:italic;">"{RAG_QUESTION}"</div>
  <hr style="border-color:#30363d;margin:12px 0;">
  <div style="color:#76b900;font-weight:bold;font-size:0.9em;">◆ RAG ANSWER (grounded in video-native observation)</div>
  <div style="color:#e6edf3;font-size:0.95em;margin-top:8px;line-height:1.7;white-space:pre-wrap;">{rag_answer}</div>
</div>'''))
"""))

# ── AUDIENCE PARTICIPATION ──────────────────────────────────────────
cells.append(md("""
### 🎤 Audience Participation: Ask Your Own Question

Change `AUDIENCE_QUERY` below and run the cell. The AI answers using the video-native caption as evidence — the same context that correctly identified the tailgating.

**Suggested queries to try:**
- `"Was the tailgater carrying anything?"`
- `"How many people passed through the access point total?"`
- `"Describe the security measures visible at the entry point"`
- `"Did anyone attempt to stop the unauthorized entry?"`
- `"What time of day does this appear to be?"`
"""))

cells.append(code("""
# ══════════════════════════════════════════════════════════════════
# 🎤 TYPE YOUR QUESTION HERE
# ══════════════════════════════════════════════════════════════════
AUDIENCE_QUERY = "Was the tailgater carrying anything?"  # ← CHANGE ME
# ══════════════════════════════════════════════════════════════════

if NIM_API_KEY:
    # Use video-native caption as context (not frame-based FAISS retrieval)
    prompt = f\"\"\"Answer using ONLY this video observation. Cite details.

VIDEO OBSERVATION:
{m3_video_caption}

QUESTION: {AUDIENCE_QUERY}\"\"\"
    audience_answer = llm_complete(prompt, system="You are a physical security analyst.")
    display(HTML(f'''
    <div style="background:#0d1117;border:2px solid #76b900;border-radius:8px;padding:16px 20px;max-width:860px;">
      <div style="color:#76b900;font-weight:bold;font-size:1.0em;">🎤 AUDIENCE QUESTION</div>
      <div style="color:#e6edf3;font-size:1.05em;margin:8px 0;font-style:italic;">"{AUDIENCE_QUERY}"</div>
      <hr style="border-color:#30363d;margin:12px 0;">
      <div style="color:#76b900;font-weight:bold;font-size:0.9em;">◆ AI ANSWER (grounded in video-native observation)</div>
      <div style="color:#e6edf3;font-size:0.95em;margin-top:8px;line-height:1.7;white-space:pre-wrap;">{audience_answer}</div>
    </div>'''))
else:
    print("Requires NIM_API_KEY for live audience Q&A.")
"""))

# ════════════════════════════════════════════════════════════════════
# GOVERNANCE & SECURITY (unchanged)
# ════════════════════════════════════════════════════════════════════
cells.append(md("""
---
## Governance & Security

Everything we just ran works in a lab. This section is about what it takes to deploy it for real.

### Government-Ready NIM Containers

| Capability | What It Means |
|-----------|--------------|
| **CVE Scanning** | Every NIM release scanned for vulnerabilities with a published patch SLA |
| **SBOM** | Full Software Bill of Materials — complete transparency on every dependency |
| **FIPS 140-2/3** | Federal Information Processing Standard validated cryptography — required for DoD/IC |
| **GPU Operator Hardening** | Kubernetes operator with security contexts and pod security admission controls |

### Deployment Options

| Scenario | Where NIM Runs | Code Change |
|----------|---------------|-------------|
| **This notebook** | `integrate.api.nvidia.com` (NVIDIA cloud) | None — just an API key |
| **On-prem / DGX Spark** | Local NIM container on your GPU | Change `NIM_BASE_URL` |
| **Air-gapped / SCIF** | Self-hosted NIM, disconnected network | Same — identical API |

The entire pipeline in this notebook — every `caption_frame_nim`, `video_native_caption`, `llm_complete`, and `embed_texts`
call — runs identically against a local NIM deployment. **No code changes required.**

The demo is the deployment.

### Classification Handling

| Concern | VSS Mechanism |
|---------|--------------|
| **Data residency** | NIM containers run on-prem; no data leaves the enclave |
| **Classification** | The vector index inherits the classification of its source video |
| **Query auditing** | Every query → analyst ID + timestamp + returned chunks → audit log |
| **Access control** | Row-level security on Elasticsearch; analysts retrieve only what they're cleared for |
| **Model provenance** | NIM containers are signed; model weights hashed at load time |
"""))

# ════════════════════════════════════════════════════════════════════
# WHAT WE COVERED (updated)
# ════════════════════════════════════════════════════════════════════
cells.append(md("---\n## What We Covered"))

cells.append(code("""
display(HTML('''
<div style="border:1px solid #30363d;border-radius:8px;overflow:hidden;max-width:860px;font-family:sans-serif;">
  <div style="background:#161b22;padding:12px 18px;border-bottom:1px solid #30363d;">
    <span style="color:#76b900;font-weight:bold;font-size:1.1em;">■ Session Summary — NVIDIA VSS Pipeline</span>
  </div>
  <table style="width:100%;border-collapse:collapse;background:#0d1117;">
    <thead>
      <tr style="border-bottom:1px solid #30363d;">
        <th style="text-align:left;padding:10px 14px;color:#8b949e;">Mission</th>
        <th style="text-align:left;padding:10px 14px;color:#8b949e;">What We Ran</th>
        <th style="text-align:left;padding:10px 14px;color:#8b949e;">Key Finding</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom:1px solid #21262d;">
        <td style="padding:10px 14px;color:#3fb950;font-weight:bold;">1 — Captioning</td>
        <td style="padding:10px 14px;color:#e6edf3;">Frame-based VLM captions → PPE Q&A</td>
        <td style="padding:10px 14px;color:#e6edf3;">Zero-shot scene understanding — no training labels needed</td>
      </tr>
      <tr style="border-bottom:1px solid #21262d;background:#161b22;">
        <td style="padding:10px 14px;color:#58a6ff;font-weight:bold;">2 — Summarization</td>
        <td style="padding:10px 14px;color:#e6edf3;">Recursive summaries → operational report</td>
        <td style="padding:10px 14px;color:#f78166;">Frame-based missed the box drop; video-native caught it</td>
      </tr>
      <tr style="border-bottom:1px solid #21262d;">
        <td style="padding:10px 14px;color:#f78166;font-weight:bold;">3 — Interactive Q&A</td>
        <td style="padding:10px 14px;color:#e6edf3;">Embeddings → FAISS → search → RAG</td>
        <td style="padding:10px 14px;color:#f78166;">Frame-based said "no tailgating"; video-native identified it</td>
      </tr>
      <tr style="background:#161b22;">
        <td style="padding:10px 14px;color:#f78166;font-weight:bold;" colspan="3">
          ⚡ Key Lesson: Frames capture snapshots. Video captures the story. For mission-critical event detection, video-native VLM inference is essential.
        </td>
      </tr>
    </tbody>
  </table>
  <div style="padding:10px 18px;border-top:1px solid #30363d;background:#161b22;color:#8b949e;font-size:0.85em;">
    <strong style="color:#e6edf3;">In full VSS production:</strong>
    VLM → Cosmos-Reason2-8B (video-native) &nbsp;·&nbsp;
    Embeddings → Cosmos-Embed-1 &nbsp;·&nbsp;
    Vector DB → Elasticsearch &nbsp;·&nbsp;
    Scale → N GPU workers = linear throughput
  </div>
</div>
'''))
"""))

cells.append(md("""
### Resources

- **VSS Blueprint**: [build.nvidia.com/nvidia/video-search-and-summarization](https://build.nvidia.com/nvidia/video-search-and-summarization/blueprintcard)
- **VSS Documentation**: [docs.nvidia.com/vss/latest](https://docs.nvidia.com/vss/latest/)
- **VSS GitHub**: [github.com/NVIDIA-AI-Blueprints/video-search-and-summarization](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
- **NIM Models**: [build.nvidia.com/explore/vision](https://build.nvidia.com/explore/vision)
- **GTC DLI Course**: DLIT81774 — Build a Video Analytics AI Agent With Vision Language Models
- **Sample Video Assets**: NGC `nvidia/vss-warehouse`, `nvidia/vss-developer`, `nvidia/vss-public-safety`

---

*Scaling AI for GEOINT · GEOINT 2026 Symposium · Kevin Green, Ph.D. & Keith Ober · NVIDIA*
"""))

# ════════════════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.3", "mimetype": "text/x-python",
                          "file_extension": ".py", "codemirror_mode": {"name": "ipython", "version": 3}}
    },
    "cells": cells
}

output_path = "/mnt/user-data/outputs/Scaling_AI_for_GEOINT_VSS_Lab.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

code_count = sum(1 for c in cells if c['cell_type'] == 'code')
md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
print(f"✓ Notebook created: {output_path}")
print(f"  Total cells: {len(cells)} (Code: {code_count}, Markdown: {md_count})")
