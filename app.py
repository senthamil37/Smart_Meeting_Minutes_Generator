import streamlit as st
import whisper
import tempfile
from transformers import pipeline

# Load Whisper model for transcription
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

whisper_model = load_whisper()
summarizer = load_summarizer()

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    result = whisper_model.transcribe(tmp_path)
    return result["text"]

def summarize_text(text):
    # Summarize in chunks if too long
    max_chunk = 800
    text = text.replace('. ', '.\n')
    sentences = text.split('\n')
    current_chunk = ''
    chunks = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk:
            current_chunk += sentence + ' '
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ' '
    chunks.append(current_chunk)
    summary = ''
    for chunk in chunks:
        if chunk.strip():
            summary += summarizer(chunk.strip())[0]['summary_text'] + ' '
    return summary.strip()

def extract_action_items(text):
    # Use a prompt-based approach for action item extraction
    prompt = f"""Extract all action items from the following meeting transcript. List them as bullet points.\n\nTranscript:\n{text}\n\nAction Items:"""
    # Use summarizer as a generic text generator
    items = summarizer(prompt, max_length=60, min_length=10, do_sample=False)[0]['summary_text']
    return items

st.title("Smart Meeting Minutes Generator 📝")
st.write("Upload a meeting audio file and get concise, structured minutes with action items!")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner('Transcribing audio...'):
        transcript = transcribe_audio(uploaded_file)
    st.subheader("Transcript")
    st.write(transcript)
    with st.spinner('Summarizing and extracting action items..."):
        summary = summarize_text(transcript)
        action_items = extract_action_items(transcript)
    st.subheader("Meeting Summary")
    st.write(summary)
    st.subheader("Action Items")
    st.write(action_items)
    # Downloadable minutes
    minutes = f"Meeting Summary:\n{summary}\n\nAction Items:\n{action_items}"
    st.download_button("Download Minutes", data=minutes, file_name="meeting_minutes.txt")
else:
    st.info("Please upload a meeting audio file to get started.") 