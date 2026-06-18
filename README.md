WebRTC AI Voice Chat
====================

Overview
--------

The goal of this project is to demo `speech <-> langchain <-> audio` workflow.

1. Speech to text is using [OpenAI's open source Whisper mini](https://huggingface.co/openai/whisper-small) model.
2. Chat model used for this demo is [Microsoft's Phi3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) model running locally using [Ollama](https://ollama.com/).
3. Text to Audio is using [Suno's open source Bark small](https://huggingface.co/suno/bark-small) model.

For interesting projects and related resources, checkout the [Awesome Projects Page](AwesomeProjects.md).

Setup
-----

This project targets Python 3.11 or newer and declares its runtime dependencies in `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Ollama with the chat model you want to use, then run the demo server:

```bash
python server.py --ollama-host http://localhost:11434
```

Open `http://localhost:8080` in a browser. Use `--cert-file` and `--key-file` if you need HTTPS.

Demo
----

Unmute the audio to hear responses  

https://github.com/lalanikarim/webrtc-ai-voice-chat/assets/1296705/7aa05d6f-ff05-4c72-b2e8-6e4e1119a68c


