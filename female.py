import os
import asyncio
import pyttsx3
import edge_tts
from pathlib import Path
import concurrent.futures
import hashlib
import shutil

# ----------------------------
# 1️⃣ Ultra-fast Local TTS (Female Voice)
# ----------------------------
def fast_local_tts(text, file_name="output", rate=200):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    voices = engine.getProperty('voices')

    # Select a female voice (Windows: Zira)
    female_voice = None
    for v in voices:
        if "female" in v.name.lower() or "zira" in v.name.lower():
            female_voice = v
            break
    if female_voice:
        engine.setProperty('voice', female_voice.id)
    else:
        engine.setProperty('voice', voices[0].id)

    engine.save_to_file(text, f"{file_name}.wav")
    engine.runAndWait()
    print(f"✅ Local TTS saved as {file_name}.wav (Female Voice)")

# ----------------------------
# 2️⃣ Microsoft Edge TTS (Female Voice)
# ----------------------------
async def fast_edge_tts(text, voice="en-US-JessaNeural", file_name="output"):
    """Edge TTS female voices: 'en-US-JessaNeural', 'en-US-AriaNeural'"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(f"{file_name}.wav")
    print(f"✅ Edge TTS saved as {file_name}.wav (Female Voice)")

def edge_tts_sync(text, voice="en-US-JessaNeural", file_name="output"):
    asyncio.run(fast_edge_tts(text, voice, file_name))

# ----------------------------
# 3️⃣ Cached TTS (Avoid Re-generation)
# ----------------------------
class CachedTTS:
    def __init__(self, cache_dir="tts_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def tts_with_cache(self, text, file_name="output"):
        cache_key = self.get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.wav"
        if cache_file.exists():
            shutil.copy(cache_file, f"{file_name}.wav")
            print(f"⚡ Using cached audio: {file_name}.wav")
            return
        fast_local_tts(text, file_name=file_name)
        shutil.copy(f"{file_name}.wav", cache_file)

# ----------------------------
# 4️⃣ Parallel Generation (Multiple TTS in threads)
# ----------------------------
def parallel_tts_generation(text_list):
    def generate_single(args):
        text, index = args
        file_name = f"output_{index}"
        fast_local_tts(text, file_name=file_name)
        return f"{file_name}.wav"

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(generate_single, [(t, i) for i, t in enumerate(text_list)]))
    print(f"✅ Generated {len(results)} audio files in parallel (Female Voice)")
    return results

# ----------------------------
# Usage Examples
# ----------------------------
if __name__ == "__main__":
    text = "Hello Bharath! This is an ultra-fast TTS solution with female voice."

    print("1️⃣ Local TTS (Female Voice):")
    fast_local_tts(text)

    print("\n2️⃣ Edge TTS (Female Voice):")
    edge_tts_sync(text)

    print("\n3️⃣ Cached TTS (Female Voice):")
    cached_tts = CachedTTS()
    cached_tts.tts_with_cache(text)

    print("\n4️⃣ Parallel Generation (Female Voice):")
    texts = ["Hello!", "How are you?", "This is fast!"]
    parallel_tts_generation(texts)
