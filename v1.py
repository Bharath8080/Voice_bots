import streamlit as st
import tempfile
import os
from pathlib import Path
import io
import base64
from groq import Groq
from together import Together
from deepgram import DeepgramClient, SpeakOptions
import time

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

# Sidebar for API keys and settings
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Keys
    groq_api_key = st.text_input("Groq API Key", type="password", key="groq_key")
    together_api_key = st.text_input("Together AI API Key", type="password", key="together_key")
    deepgram_api_key = st.text_input("Deepgram API Key", type="password", key="deepgram_key")
    
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
        "Deepgram TTS Model",
        ["aura-2-zeus-en", "aura-2-thalia-en", "aura-2-luna-en", "aura-2-stella-en"],
        index=0
    )
    
    st.divider()
    
    # Voice Settings
    st.subheader("🔊 Voice Settings")
    auto_play_tts = st.checkbox("Auto-play TTS", value=True)
    speech_language = st.selectbox(
        "Speech Language",
        ["en", "es", "fr", "de", "ja", "zh"],
        index=0
    )
    
    st.divider()
    
    # Clear Chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main app header
st.title("🎤 Voice Chatbot")
st.markdown("Speak, and I'll respond with both text and voice!")

# Check if all API keys are provided
api_keys_provided = all([groq_api_key, together_api_key, deepgram_api_key])

if not api_keys_provided:
    st.warning("⚠️ Please provide all API keys in the sidebar to use the voice chatbot.")
    st.info("""
    **Required API Keys:**
    - **Groq API Key**: For LLM responses
    - **Together AI API Key**: For speech-to-text transcription  
    - **Deepgram API Key**: For text-to-speech synthesis
    """)
    st.stop()

# Initialize clients
try:
    groq_client = Groq(api_key=groq_api_key)
    together_client = Together(api_key=together_api_key)
    deepgram_client = DeepgramClient(deepgram_api_key)
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

def get_ai_response(messages):
    """Get AI response from Groq"""
    try:
        with st.spinner("🤖 Generating response..."):
            response = groq_client.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                stream=False
            )
            return response.choices[0].message.content
    except Exception as e:
        st.error(f"AI response error: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech using Deepgram"""
    try:
        with st.spinner("🔊 Generating speech..."):
            options = SpeakOptions(model=tts_model)
            
            # Use a unique filename to avoid conflicts
            temp_filename = f"temp_audio_{int(time.time() * 1000)}.mp3"
            temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
            
            response = deepgram_client.speak.rest.v("1").save(
                temp_path,
                {"text": text},
                options
            )
            
            # Small delay to ensure file is written
            time.sleep(0.1)
            
            # Read the audio file
            audio_data = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(temp_path, "rb") as audio_file:
                        audio_data = audio_file.read()
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.2)
                    else:
                        raise
            
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except PermissionError:
                # If we can't delete it immediately, it will be cleaned up later
                pass
                
            return audio_data
    except Exception as e:
        st.error(f"TTS error: {str(e)}")
        return None

def display_chat_message(role, content, audio_data=None):
    """Display a chat message with optional audio"""
    with st.chat_message(role):
        st.write(content)
        if audio_data and role == "assistant":
            st.audio(audio_data, format="audio/mp3", autoplay=auto_play_tts)

# Display existing messages
for message in st.session_state.messages:
    display_chat_message(
        message["role"], 
        message["content"], 
        message.get("audio_data")
    )

# Audio input section
st.divider()
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
                for msg in st.session_state.messages[-10:]  # Last 10 messages for context
            ])
            
            # Get AI response
            ai_response = get_ai_response(messages_for_ai)
            
            if ai_response:
                # Generate TTS audio
                audio_data = text_to_speech(ai_response)
                
                # Add assistant message to chat
                assistant_message = {
                    "role": "assistant", 
                    "content": ai_response,
                    "audio_data": audio_data
                }
                st.session_state.messages.append(assistant_message)
                
                # Display assistant message
                display_chat_message("assistant", ai_response, audio_data)
                
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    
    finally:
        st.session_state.is_processing = False
        st.rerun()

# Text input fallback
st.divider()

# Use a form to handle text input properly
with st.form("text_input_form", clear_on_submit=True):
    text_input = st.text_input(
        "💬 Or type your message:", 
        disabled=st.session_state.is_processing,
        key="text_message"
    )
    text_submit = st.form_submit_button("Send")

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
            for msg in st.session_state.messages[-10:]  # Last 10 messages for context
        ])
        
        # Get AI response
        ai_response = get_ai_response(messages_for_ai)
        
        if ai_response:
            # Generate TTS audio
            audio_data = text_to_speech(ai_response)
            
            # Add assistant message to chat
            assistant_message = {
                "role": "assistant", 
                "content": ai_response,
                "audio_data": audio_data
            }
            st.session_state.messages.append(assistant_message)
            
            # Display assistant message
            display_chat_message("assistant", ai_response, audio_data)
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    
    finally:
        st.session_state.is_processing = False
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🎤 Voice Chatbot powered by Groq LLM, Together AI STT, and Deepgram TTS
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("📖 How to use"):
    st.markdown("""
    **Voice Mode:**
    1. Click the microphone button and record your message
    2. Click "Process" to transcribe and get AI response
    3. Listen to the AI's voice response (auto-play enabled by default)
    
    **Text Mode:**
    1. Type your message in the text input
    2. Press Enter to get AI response with voice
    
    **Settings:**
    - Configure API keys in the sidebar
    - Choose different models for better results
    - Adjust language settings for transcription
    - Toggle auto-play for TTS responses
    
    **Tips:**
    - Speak clearly for better transcription accuracy
    - Use the language selector for non-English audio
    - Clear chat history using the sidebar button
    """)

# Status indicator
if st.session_state.is_processing:
    st.info("🔄 Processing your request...")
