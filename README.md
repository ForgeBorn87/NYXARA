# NYXARA

A personality-driven AI chat assistant powered by Ministral 3B.

## Features
- Local LLM inference (Ministral 3B)
- Text-to-speech with Edge TTS
- Tool modes: Clarity, Focus, Feedback, Analyze
- Session-based conversation history
- Dark themed web interface

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python nyxara.py
```

Then open http://localhost:5001

## Requirements
- Python 3.10+
- ~8GB VRAM for model inference
- CUDA-capable GPU recommended
