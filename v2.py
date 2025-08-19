import streamlit as st
import tempfile
import os
from pathlib import Path
import io
import base64
from groq import Groq
from together import Together
from openai import OpenAI
import time
from dotenv import load_dotenv
import wave
import numpy as np

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🎤 Voice Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "voice"  # "voice" or "text"
if "streaming_response" not in st.session_state:
    st.session_state.streaming_response = ""

# Load API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

# Sidebar for settings
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key Status
    st.subheader("🔑 API Keys Status")
    if GROQ_API_KEY:
        st.success("✅ Groq")
    else:
        st.error("❌ Groq")
        
    if TOGETHER_API_KEY:
        st.success("✅ Together AI") 
    else:
        st.error("❌ Together AI")
        
    if DEEPINFRA_API_KEY:
        st.success("✅ DeepInfra")
    else:
        st.error("❌ DeepInfra")
    
    if not all([GROQ_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY]):
        st.warning("⚠️ Add missing API keys to your .env file")
    
    st.divider()
    
    # Model Settings
    st.subheader("🎯 Model Settings")
    groq_model = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    
    whisper_model = st.selectbox(
        "Together AI Whisper Model",
        ["openai/whisper-large-v3", "openai/whisper-large-v2"],
        index=0
    )
    
    tts_model = st.selectbox(
        "DeepInfra TTS Model",
        ["hexgrad/Kokoro-82M"],
        index=0,
        help="High-quality neural TTS model"
    )
    
    st.divider()
    
    # Voice Settings
    st.subheader("🔊 Voice Settings")
    auto_play_tts = st.checkbox("Auto-play TTS", value=True)
    
    speech_language = st.selectbox(
        "Speech Language",
        ["en", "es", "fr", "de", "ja", "zh", "hi", "it", "ko", "pt", "ru", "ar"],
        index=0
    )
    
    # Voice selection for TTS
    tts_voice = st.selectbox(
        "TTS Voice",
        [
            "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael",
            "bf_emma", "bf_isabella", "bm_george", "bm_lewis", "ef_anna", "ef_emma", 
            "ef_isabella", "em_ben", "em_daniel", "em_william"
        ],
        index=0,
        help="Choose from available DeepInfra voices"
    )
    
    audio_format = st.selectbox(
        "Audio Format",
        ["mp3", "opus", "pcm", "flac", "wav"],
        index=0,
        help="PCM for lowest latency, MP3 for general use"
    )
    
    st.divider()
    
    # Clear Chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.streaming_response = ""
        st.rerun()

# Main app header
st.title("🎤 Voice Chatbot")
st.markdown("Speak or type, and I'll respond with live streaming and high-quality TTS!")

# Check if all API keys are provided
if not all([GROQ_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY]):
    st.error("❌ Missing API keys! Please add them to your .env file:")
    st.code("""
GROQ_API_KEY=your_groq_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
DEEPINFRA_API_KEY=your_deepinfra_api_key_here
    """)
    st.stop()

# Initialize clients
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    together_client = Together(api_key=TOGETHER_API_KEY)
    
    # Initialize DeepInfra client
    deepinfra_client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=DEEPINFRA_API_KEY
    )
except Exception as e:
    st.error(f"Error initializing clients: {str(e)}")
    st.stop()

def transcribe_audio(audio_file, language="en"):
    """Transcribe audio using Together AI Whisper"""
    try:
        with st.spinner("🎯 Transcribing audio..."):
            response = together_client.audio.transcriptions.create(
                file=audio_file,
                model=whisper_model,
                language=language
            )
            return response.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def stream_ai_response(messages, placeholder):
    """Stream AI response from Groq with live updates"""
    try:
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.write(full_response + "▌")
        
        placeholder.write(full_response)
        return full_response
    except Exception as e:
        st.error(f"AI response error: {str(e)}")
        return None

def text_to_speech_deepinfra(text):
    """Convert text to speech using DeepInfra TTS API"""
    try:
        with st.spinner("🔊 Generating speech with DeepInfra..."):
            # Create temporary file path
            temp_filename = f"speech_{int(time.time() * 1000)}.{audio_format}"
            temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
            
            # Generate speech using DeepInfra
            with deepinfra_client.audio.speech.with_streaming_response.create(
                model=tts_model,
                voice=tts_voice,
                input=text,
                response_format=audio_format,
            ) as response:
                response.stream_to_file(temp_path)
            
            # Read the generated audio file
            with open(temp_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except PermissionError:
                pass
                
            return audio_data
            
    except Exception as e:
        st.error(f"DeepInfra TTS error: {str(e)}")
        return None

def display_chat_message(role, content, audio_data=None):
    """Display a chat message with optional audio"""
    with st.chat_message(role):
        st.write(content)
        if audio_data and role == "assistant":
            # Determine audio format for st.audio
            format_mapping = {
                "mp3": "audio/mp3",
                "wav": "audio/wav", 
                "opus": "audio/ogg",
                "flac": "audio/flac",
                "pcm": "audio/wav"
            }
            audio_mime = format_mapping.get(audio_format, "audio/mp3")
            st.audio(audio_data, format=audio_mime, autoplay=auto_play_tts)

# Input mode toggle buttons
st.subheader("🎛️ Input Mode")
col1, col2 = st.columns(2)

with col1:
    if st.button(
        "🎤 Voice Input", 
        type="primary" if st.session_state.input_mode == "voice" else "secondary",
        use_container_width=True
    ):
        st.session_state.input_mode = "voice"
        st.rerun()

with col2:
    if st.button(
        "⌨️ Text Input", 
        type="primary" if st.session_state.input_mode == "text" else "secondary",
        use_container_width=True
    ):
        st.session_state.input_mode = "text"
        st.rerun()

st.divider()

# Display existing messages
for message in st.session_state.messages:
    display_chat_message(
        message["role"], 
        message["content"], 
        message.get("audio_data")
    )

# Input based on selected mode
if st.session_state.input_mode == "voice":
    st.subheader("🎤 Voice Input Mode")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        audio_input = st.audio_input("🎤 Record your message")
    
    with col2:
        st.write("")  # Spacer
        process_audio = st.button(
            "🚀 Process", 
            disabled=not audio_input or st.session_state.is_processing,
            use_container_width=True
        )
    
    # Process audio when button is clicked
    if process_audio and audio_input and not st.session_state.is_processing:
        st.session_state.is_processing = True
        
        try:
            # Save audio to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_input.read())
                tmp_file_path = tmp_file.name
            
            # Transcribe audio
            transcript = transcribe_audio(tmp_file_path, speech_language)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            if transcript:
                # Add user message to chat
                user_message = {"role": "user", "content": transcript}
                st.session_state.messages.append(user_message)
                
                # Display user message
                display_chat_message("user", transcript)
                
                # Prepare messages for AI
                messages_for_ai = [{"role": "system", "content": "You are a helpful AI assistant. Provide concise, friendly responses."}]
                messages_for_ai.extend([
                    {"role": msg["role"], "content": msg["content"]} 
                    for msg in st.session_state.messages[-10:]
                ])
                
                # Create placeholder for streaming response
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    
                    # Stream AI response
                    ai_response = stream_ai_response(messages_for_ai, response_placeholder)
                    
                    if ai_response:
                        # Generate TTS audio
                        audio_data = text_to_speech_deepinfra(ai_response)
                        
                        # Add assistant message to chat
                        assistant_message = {
                            "role": "assistant", 
                            "content": ai_response,
                            "audio_data": audio_data
                        }
                        st.session_state.messages.append(assistant_message)
                        
                        # Display audio
                        if audio_data:
                            format_mapping = {
                                "mp3": "audio/mp3",
                                "wav": "audio/wav", 
                                "opus": "audio/ogg",
                                "flac": "audio/flac",
                                "pcm": "audio/wav"
                            }
                            audio_mime = format_mapping.get(audio_format, "audio/mp3")
                            st.audio(audio_data, format=audio_mime, autoplay=auto_play_tts)
                        
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        
        finally:
            st.session_state.is_processing = False
            st.rerun()

else:  # Text input mode
    st.subheader("⌨️ Text Input Mode")
    
    # Use a form to handle text input properly
    with st.form("text_input_form", clear_on_submit=True):
        text_input = st.text_area(
            "💬 Type your message:", 
            disabled=st.session_state.is_processing,
            key="text_message",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            text_submit = st.form_submit_button(
                "🚀 Send", 
                use_container_width=True,
                type="primary"
            )
    
    if text_submit and text_input and not st.session_state.is_processing:
        st.session_state.is_processing = True
        
        try:
            # Add user message to chat
            user_message = {"role": "user", "content": text_input}
            st.session_state.messages.append(user_message)
            
            # Display user message
            display_chat_message("user", text_input)
            
            # Prepare messages for AI
            messages_for_ai = [{"role": "system", "content": "You are a helpful AI assistant. Provide concise, friendly responses."}]
            messages_for_ai.extend([
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.messages[-10:]
            ])
            
            # Create placeholder for streaming response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                # Stream AI response
                ai_response = stream_ai_response(messages_for_ai, response_placeholder)
                
                if ai_response:
                    # Generate TTS audio
                    audio_data = text_to_speech_deepinfra(ai_response)
                    
                    # Add assistant message to chat
                    assistant_message = {
                        "role": "assistant", 
                        "content": ai_response,
                        "audio_data": audio_data
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Display audio
                    if audio_data:
                        format_mapping = {
                            "mp3": "audio/mp3",
                            "wav": "audio/wav", 
                            "opus": "audio/ogg",
                            "flac": "audio/flac",
                            "pcm": "audio/wav"
                        }
                        audio_mime = format_mapping.get(audio_format, "audio/mp3")
                        st.audio(audio_data, format=audio_mime, autoplay=auto_play_tts)
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        
        finally:
            st.session_state.is_processing = False
            st.rerun()

# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🎤 Voice Chatbot | Mode: {st.session_state.input_mode.title()} | Powered by Groq, Together AI & DeepInfra TTS
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("📖 How to use"):
    st.markdown("""
    **Setup:**
    1. Create a `.env` file in your project root with your API keys:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    TOGETHER_API_KEY=your_together_api_key_here
    DEEPINFRA_API_KEY=your_deepinfra_api_key_here
    ```
    
    **Voice Mode:**
    1. Click "🎤 Voice Input" button to switch to voice mode
    2. Record your message using the microphone
    3. Click "Process" to get live streaming AI response with voice
    
    **Text Mode:**
    1. Click "⌨️ Text Input" button to switch to text mode
    2. Type your message in the text area
    3. Click "Send" to get live streaming AI response with voice
    
    **Features:**
    - 🔄 **Live Streaming**: Responses stream in real-time as they're generated
    - 🎛️ **Mode Toggle**: Switch between voice and text input with buttons
    - 🔑 **Auto API Keys**: Automatically loads keys from .env file
    - 🎭 **16 Voices**: Choose from DeepInfra's high-quality voice collection
    - 🎨 **Audio Formats**: PCM for lowest latency, MP3/Opus for quality
    - 💬 **Chat History**: Maintains conversation context
    - 🌍 **12+ Languages**: Automatic language detection and support
    
    **Available Voices:**
    - **Female**: af_bella, af_nicole, af_sarah, af_sky, bf_emma, bf_isabella, ef_anna, ef_emma, ef_isabella
    - **Male**: am_adam, am_michael, bm_george, bm_lewis, em_ben, em_daniel, em_william
    
    **Audio Format Guide:**
    - **PCM**: Fastest response, lowest latency
    - **MP3**: Balanced quality and file size
    - **Opus**: Best for streaming
    - **FLAC**: Highest quality (larger files)
    - **WAV**: Uncompressed, high quality
    
    **Tips:**
    - Speak clearly for better transcription accuracy
    - Use PCM format for fastest audio generation
    - Try different voices to find your preference
    - Clear chat history using the sidebar button
    """)

# Status indicator
if st.session_state.is_processing:
    st.info(f"🔄 Processing your {st.session_state.input_mode} input...")
