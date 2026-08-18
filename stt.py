"""
Speech-to-text using Sarvam AI (Saaras v3).
Setup:
    1. Get an API key from https://dashboard.sarvam.ai
    2. Set it as an environment variable (don't hardcode it):
         PowerShell:  $env:SARVAM_API_KEY = "your_key_here"
    3. pip install requests
Usage:
    from stt import transcribe
    text = transcribe("my_recording.wav")
    print(text)
"""
import os
import requests

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


def transcribe(audio_file_path: str, mode: str = "transcribe", retries: int = 2) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY environment variable not set. "
            "Run: $env:SARVAM_API_KEY = 'your_key_here' in PowerShell before starting."
        )
    headers = {"api-subscription-key": SARVAM_API_KEY}
    data = {"model": "saaras:v3", "mode": mode}
    last_error = None
    for attempt in range(retries + 1):
        try:
            with open(audio_file_path, "rb") as f:
                files = {"file": (os.path.basename(audio_file_path), f, "audio/wav")}
                response = requests.post(
                    SARVAM_STT_URL,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=30,
                )
            if response.status_code >= 400:
                print(f"[transcribe] server said: {response.status_code} {response.text}")
            response.raise_for_status()
            result = response.json()
            return result.get("transcript", "")
        except Exception as e:
            last_error = e
            print(f"[transcribe] attempt {attempt + 1} failed: {e}")
    print(f"[transcribe] giving up after {retries + 1} attempts: {last_error}")
    return ""


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stt.py <audio_file_path>")
    else:
        text = transcribe(sys.argv[1])
        print("Transcribed text:", text)