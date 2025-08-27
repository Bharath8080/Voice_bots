# Simple Gemini TTS Generator
# Install: pip install google-genai

import os
import mimetypes
import struct
from google import genai
from google.genai import types


def save_audio(file_name, audio_data, mime_type):
    """Save audio data to file (default .wav if not detected)."""
    file_extension = mimetypes.guess_extension(mime_type)
    if file_extension is None:
        file_extension = ".wav"
        audio_data = convert_to_wav(audio_data, mime_type)
    with open(f"{file_name}{file_extension}", "wb") as f:
        f.write(audio_data)
    print(f"✅ Audio saved as {file_name}{file_extension}")


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Add WAV header for raw audio data if needed."""
    bits_per_sample = 16
    sample_rate = 24000
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE",
        b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", data_size
    )
    return header + audio_data


def tts(text, voice="Zephyr", file_name="output"):
    """Generate TTS from Gemini and save as audio file."""
    client = genai.Client(api_key="AIzaSyD0QR5NUeACg8lqkQKZWyFigAJYbFt0BeI")


    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            ),
        ),
    )

    part = response.candidates[0].content.parts[0].inline_data
    save_audio(file_name, part.data, part.mime_type)


if __name__ == "__main__":
    # Example usage
    tts("Hello Bharath! This is your simple TTS generator. How are you?", voice="Zephyr", file_name="my_audio3")
