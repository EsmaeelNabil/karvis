```bash
uv venv
source .venv/bin/activate

uv pip install -r requirements.txt
uv run main.py
```

### Configuration with LLM

```python
# --- Configuration ---
TRANSCRIBER_MODEL = "moonshine/base"
SILENCE_TIMEOUT_MS = 300
# LLM_MODEL = "gpt-4o-mini"
LLM_MODEL = "cogito:3b"
USE_LOCAL_LLM = True
TTS_MODEL_ID = "prince-canuma/Kokoro-82M"
TTS_VOICE = "af_bella"
```
