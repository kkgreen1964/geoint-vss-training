# geoint-vss-training

**Scaling AI for GEOINT: Video Analytics with NVIDIA Vision Language Models**

Kevin Green, Ph.D. — NVIDIA WWFO | April 2026

---

## Overview

This repository contains the interactive lab materials for the GEOINT 2026 Symposium training session:

> **Scaling AI for GEOINT: Practical Foundations with NVIDIA Technologies**
>
> Tuesday, May 5, 2026 | 2:00 to 3:00 PM MST | Aurora, Colorado
>
> Speakers: Kevin Green, Ph.D. and Dr. Aaron Reite

The lab demonstrates how to build video analytics AI agents using NVIDIA's Vision Language Models (VLMs) and Large Language Models (LLMs) through the [NVIDIA Blueprint for Video Search and Summarization (VSS)](https://build.nvidia.com/nvidia/video-search-and-summarization/blueprintcard). It is structured as three progressive "missions" that walk participants from basic VLM captioning through recursive video summarization to interactive agentic Q&A, all powered by [NVIDIA NIM](https://build.nvidia.com/explore/vision) microservices.

### Key Technologies

- **Cosmos Reason 2 8B** — NVIDIA's open reasoning VLM for physical AI with enhanced spatial-temporal understanding
- **Nemotron Nano 9B v2** — Efficient LLM optimized for tool calling, structured output, and reasoning
- **NVIDIA NIM** — Pre-optimized inference microservices accessible via OpenAI-compatible APIs
- **NVIDIA Blueprint for VSS** — Reference architecture for building video analytics AI agents

## Three Missions

| Mission | Video | What It Demonstrates | Key Insight |
|---------|-------|---------------------|-------------|
| **1 — Captioning** | `warehouse_safety_0002.mp4` | Frame extraction, VLM dense captioning, zero-shot PPE Q&A | No training labels needed for scene understanding |
| **2 — Summarization** | `warehouse_short.mp4` | Recursive chunk summarization, operational report generation | Frame-based analysis missed the box drop; video-native caught it |
| **3 — Interactive Q&A** | `test1.mp4` (tailgating) | Embedding generation, FAISS similarity search, RAG-grounded Q&A | Frame-based said "no tailgating"; video-native identified it |

Each mission builds on the previous one, progressively demonstrating more sophisticated video understanding capabilities while highlighting the critical difference between frame-based and video-native VLM inference.

## Repository Structure

```
geoint-vss-training/
├── Scaling_AI_for_GEOINT_VSS_Lab.ipynb      # Main lab notebook (run this)
├── build_vss_lab_notebook.py                 # Generator script for reproducible notebook builds
├── data/
│   ├── videos/                              # Mission video files (see Setup)
│   │   ├── warehouse_safety_0002.mp4        # Mission 1: PPE and safety monitoring
│   │   ├── warehouse_short.mp4              # Mission 2: Warehouse operations
│   │   └── test1.mp4                        # Mission 3: Facility access / tailgating
│   ├── captions/                            # Cached VLM captions and LLM outputs (auto-generated)
│   ├── embeddings/                          # Cached FAISS embeddings (auto-generated)
│   └── frames/                             # Extracted video frames (auto-generated)
├── Scaling_AI_for_GEOINT_VSS_Lab.html       # Static HTML backup for demo resilience
└── README.md
```

## Prerequisites

### Software

- Python 3.10+
- ffmpeg (for frame extraction)
- Jupyter Lab or Jupyter Notebook
- Internet connection (for NIM API calls on first run; cached thereafter)

### Python Packages

```bash
pip install jupyter openai numpy matplotlib Pillow faiss-cpu
```

### NVIDIA NIM API Key

A free NIM API key is required to access the hosted VLM and LLM endpoints. No local GPU is needed.

1. Sign up for the [NVIDIA Developer Program](https://developer.nvidia.com/) (free)
2. Generate an API key at [build.nvidia.com](https://build.nvidia.com/settings/api-keys)
3. Set the key as an environment variable:

```bash
export NIM_API_KEY="nvapi-your-key-here"
```

Or configure it in `~/.secrets` for persistence:

```bash
echo "export NIM_API_KEY='nvapi-your-key-here'" >> ~/.secrets
chmod 600 ~/.secrets
source ~/.secrets
```

### Video Assets

The three mission videos are sourced from NVIDIA's NGC sample data. Download them using the NGC CLI:

```bash
# Install NGC CLI if needed
# https://org.ngc.nvidia.com/setup/api-key

# Download VSS sample data
ngc registry resource download-version nvidia/vss-developer/dev-profile-sample-data:3.1.0
ngc registry resource download-version nvidia/metropolis/public-safety-metropolis-app-data-tailgate:3.0.0

# Copy videos to the project
cp dev-profile-sample-data_v3.1.0/dev-profile-sample-data/warehouse_short.mp4 data/videos/
cp dev-profile-sample-data_v3.1.0/dev-profile-sample-data/warehouse_safety_0002.mp4 data/videos/
cp public-safety-metropolis-app-data-tailgate_v3.0.0/metropolis-apps-data/videos/tailgating/test1.mp4 data/videos/
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/kkgreen1964/geoint-vss-training.git
cd geoint-vss-training

# Create data directories
mkdir -p data/{videos,captions,embeddings,frames}

# Download and place video assets (see Prerequisites above)

# Set your NIM API key
export NIM_API_KEY="nvapi-your-key-here"

# Launch Jupyter
jupyter lab Scaling_AI_for_GEOINT_VSS_Lab.ipynb
```

Run all cells top to bottom. The first run calls live NIM APIs and caches every result to `data/captions/` and `data/embeddings/`. Subsequent runs load entirely from cache with no network calls required.

## Demo Resilience

The notebook is designed for reliable conference presentation with three tiers of fallback:

| Scenario | What to Do |
|----------|-----------|
| **Normal demo** | Open notebook, Shift+Enter through all cells — loads from cache instantly |
| **Re-run a specific mission** | Set `FORCE_LIVE_M1/M2/M3 = True` at the top of that mission's pipeline cell, re-run |
| **Jupyter crashes** | Open `Scaling_AI_for_GEOINT_VSS_Lab.html` in any browser — full outputs baked in |
| **Totally fresh run** | Uncomment `CLEAR_CACHE = True` at top, re-run everything |
| **No internet at venue** | Everything runs from cache — no NIM API calls needed |

### Exporting the HTML Backup

After a clean end-to-end run:

```bash
jupyter nbconvert --to html Scaling_AI_for_GEOINT_VSS_Lab.ipynb
```

This produces a self-contained HTML file (~1 MB) with all outputs rendered, your insurance policy if Jupyter is unavailable on stage.

## How It Works

### Architecture

The notebook calls NVIDIA's hosted NIM endpoints at `integrate.api.nvidia.com`. All heavy GPU compute (VLM captioning, LLM summarization, embedding generation) runs on NVIDIA's infrastructure. The local machine only needs Python, an internet connection, and about 50 MB of disk space for videos and cached results.

```
┌─────────────────────┐         ┌──────────────────────────────┐
│   Your Laptop       │  REST   │  NVIDIA build.nvidia.com     │
│   (WSL2 / macOS)    │◄──────► │                              │
│                     │  API    │  Cosmos Reason 2 8B (VLM)    │
│  Jupyter Notebook   │         │  Nemotron Nano 9B v2 (LLM)  │
│  Cached JSON/NPY    │         │  NIM Embedding Model         │
└─────────────────────┘         └──────────────────────────────┘
```

No local GPU, Docker, or VSS stack deployment required.

### Caching System

Every NIM API call is wrapped with a cache-first pattern:

1. Check if `data/captions/<result>.json` or `data/embeddings/<result>.npy` exists
2. If cached, load instantly (no API call)
3. If not cached, call the NIM API, save the result, then return

This ensures the demo runs identically whether connected to the internet or completely offline. The `FORCE_LIVE_M1/M2/M3` toggles let you selectively clear and re-run any mission's cache during the presentation.

### NIM Models Used

| Model | Role | Endpoint |
|-------|------|----------|
| `nvidia/cosmos-reason2-8b` | VLM: image/video captioning, scene understanding | `integrate.api.nvidia.com/v1` |
| `nvidia/nvidia-nemotron-nano-9b-v2` | LLM: summarization, structured output, Q&A | `integrate.api.nvidia.com/v1` |
| `nvidia/nv-embedqa-e5-v5` | Embedding: semantic similarity search | `integrate.api.nvidia.com/v1` |

All three expose OpenAI-compatible APIs.

## GEOINT 2026 Session Context

This lab accompanies a one-hour training session structured in three parts:

1. **The Foundational Infrastructure** (15 min, slides) — NVIDIA AI Factory, Five Layer Cake, Triton Inference Server, Multi-Instance GPU
2. **Deep Dive and Demo: Video Search and Summarization** (30 min, this notebook) — Three missions demonstrating VLM captioning, recursive summarization, and agentic Q&A
3. **Governance and Security** (5 min, slides) — Government-ready deployments, NVIDIA AI Enterprise, compliance

The notebook is designed for presenter-led walkthrough during Part 2, with cached results ensuring smooth execution regardless of venue connectivity.

### Related Resources

- [NVIDIA AI Blueprint for VSS](https://build.nvidia.com/nvidia/video-search-and-summarization/blueprintcard)
- [VSS Documentation](https://docs.nvidia.com/vss/latest/)
- [VSS GitHub Repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
- [Cosmos Reason 2 on build.nvidia.com](https://build.nvidia.com/nvidia/cosmos-reason2-8b)
- [Nemotron Nano 9B on build.nvidia.com](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2)
- [NVIDIA Metropolis Platform](https://www.nvidia.com/en-us/autonomous-machines/intelligent-video-analytics-platform/)
- [VLMs as NVIDIA NIM](https://build.nvidia.com/explore/vision)
- [GTC DLI Course DLIT81774](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81774/) — Full 105-minute hands-on version (Instructor: Sammy Ochoa, Lead TA: Kevin Green)

## VSS Launchable Deployment

For users who want to run the full VSS 3 stack (including LVS microservice, VIOS, ElasticSearch, and MCP-based agent), two Brev Launchable options are available:

| Launchable | GPU | Cost | Best For |
|-----------|-----|------|----------|
| [Workshop VSS](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3AuTjTao5gelkXaCcUkXTRNbdyL) | 4x L40S | 12.66 credits/hr | Hands-on labs, Jupyter pre-configured |
| [VSS 3.1 Blueprint](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-30hFiArE1zpZSEwj8PrfBykHBav) | 2x RTX PRO 6000 | 9.98 credits/hr | Full Blueprint deployment |

Request Brev credits through the `#brev` Slack channel using the credit request form. 1 Brev credit = $1.

## Author

**Kevin Green, Ph.D.**
Senior Solutions Architect, NVIDIA World-Wide Field Organization (WWFO)
Public Sector AI/ML and Computer Vision

## License

Video assets are subject to NVIDIA's [AI Foundation Models Community License](https://developer.nvidia.com/ai-foundation-models-community-license). Notebook code is provided for educational and demonstration purposes.
