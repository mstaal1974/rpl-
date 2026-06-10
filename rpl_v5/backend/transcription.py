"""
Audio/video → text transcription via Google Cloud Speech-to-Text.

Optional feature: every prerequisite (the google-cloud-speech package, ffmpeg, the
Speech-to-Text API, and — for long recordings — a GCS bucket) is checked at call
time and surfaced as a clear, actionable RuntimeError rather than a crash, so the
rest of the app runs fine without it.

ffmpeg normalises ANY input (mp3 / m4a / mp4 / wav / ogg / webm) to FLAC mono
16 kHz — this both extracts audio from video and sidesteps Speech-to-Text's
format/sample-rate detection.

Operator setup:
  * Enable the Cloud Speech-to-Text API on the project.
  * Grant the Cloud Run runtime service account roles/speech.client.
  * For recordings longer than ~1 minute, set RPL_AUDIO_BUCKET to a GCS bucket the
    SA can read/write (roles/storage.objectAdmin) — long audio must go via GCS.
  * Optional: RPL_STT_LANGUAGE (default en-AU).
"""
import os
import uuid
import shutil
import asyncio
import logging
import tempfile
import subprocess

logger = logging.getLogger(__name__)

AUDIO_BUCKET   = os.getenv("RPL_AUDIO_BUCKET", "")
STT_LANGUAGE   = os.getenv("RPL_STT_LANGUAGE", "en-AU")
SYNC_MAX_BYTES = 10 * 1024 * 1024   # inline sync recognise limit (~10 MB FLAC)


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _to_flac(src_path: str) -> str:
    """Extract/normalise audio to FLAC mono 16 kHz. Returns the output path."""
    out = src_path + ".flac"
    cmd = ["ffmpeg", "-y", "-i", src_path,
           "-vn",            # drop any video stream
           "-ac", "1",       # mono
           "-ar", "16000",   # 16 kHz
           "-c:a", "flac", out]
    proc = subprocess.run(cmd, capture_output=True, timeout=900)
    if proc.returncode != 0 or not os.path.exists(out):
        tail = (proc.stderr or b"").decode("utf-8", "ignore")[-400:]
        raise RuntimeError(f"Could not decode the recording's audio (ffmpeg): {tail}")
    return out


async def transcribe_audio(content: bytes, filename: str) -> dict:
    """
    Transcribe an uploaded audio/video file.
    Returns {transcript, method}. Raises RuntimeError with a clear message if a
    prerequisite is missing.
    """
    try:
        from google.cloud import speech  # noqa: F401
    except Exception:
        raise RuntimeError(
            "Speech-to-Text isn't available — the google-cloud-speech package isn't "
            "installed or the Speech-to-Text API isn't enabled on the project.")
    if not _have_ffmpeg():
        raise RuntimeError(
            "ffmpeg isn't installed in the container — it's required to decode/extract "
            "the recording's audio.")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_sync, content, filename)


def _transcribe_sync(content: bytes, filename: str) -> dict:
    from google.cloud import speech

    tmpdir = tempfile.mkdtemp(prefix="rpl_audio_")
    try:
        src = os.path.join(tmpdir, "input_" + os.path.basename(filename or "audio"))
        with open(src, "wb") as f:
            f.write(content)
        flac = _to_flac(src)
        with open(flac, "rb") as f:
            flac_bytes = f.read()

        client = speech.SpeechClient()
        diarization = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True, min_speaker_count=2, max_speaker_count=4)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            audio_channel_count=1,
            language_code=STT_LANGUAGE,
            enable_automatic_punctuation=True,
            diarization_config=diarization,
            model="latest_long")

        if len(flac_bytes) <= SYNC_MAX_BYTES and not AUDIO_BUCKET:
            audio = speech.RecognitionAudio(content=flac_bytes)
            response = client.recognize(config=config, audio=audio)
            method = "sync"
        else:
            if not AUDIO_BUCKET:
                raise RuntimeError(
                    "This recording is too long for inline transcription — set "
                    "RPL_AUDIO_BUCKET (a GCS bucket the service account can use) to "
                    "enable long-audio transcription.")
            from google.cloud import storage
            gcs = storage.Client()
            blob_name = f"rpl-stt/{uuid.uuid4().hex}.flac"
            blob = gcs.bucket(AUDIO_BUCKET).blob(blob_name)
            blob.upload_from_filename(flac)
            try:
                audio = speech.RecognitionAudio(uri=f"gs://{AUDIO_BUCKET}/{blob_name}")
                operation = client.long_running_recognize(config=config, audio=audio)
                response = operation.result(timeout=1800)
                method = "long_running"
            finally:
                try:
                    blob.delete()
                except Exception:
                    pass

        transcript = _format_diarized(response) or " ".join(
            r.alternatives[0].transcript.strip()
            for r in response.results if r.alternatives)
        return {"transcript": transcript.strip(), "method": method}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _format_diarized(response) -> str:
    """
    Build 'Speaker N: ...' lines from Speech-to-Text speaker diarization. The
    word-level speaker tags are populated on the LAST result's alternative.
    Returns '' if no diarization info is present.
    """
    words = []
    for r in response.results:
        if r.alternatives and r.alternatives[0].words:
            words = r.alternatives[0].words   # last result holds the full tagged list
    if not words:
        return ""
    lines, current, buf = [], None, []
    for w in words:
        tag = getattr(w, "speaker_tag", 0) or 0
        if current is not None and tag != current and buf:
            lines.append(f"Speaker {current}: " + " ".join(buf))
            buf = []
        current = tag
        buf.append(w.word)
    if buf:
        lines.append(f"Speaker {current}: " + " ".join(buf))
    return "\n".join(lines)
