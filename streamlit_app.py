import streamlit as st
import torch
from transformers import pipeline

st.set_page_config(
    page_title="Meeting Transcript Summarizer",
    layout="wide"
)

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="google/flan-t5-base",
        device=0 if torch.cuda.is_available() else -1
    )

summarizer = load_model()

def summarize_meeting(transcript):
    if len(transcript.strip()) < 50:
        return "Please provide a longer meeting transcript."

    prompt = f"""
You are an assistant that summarizes meeting transcripts.
Only use the information provided in the transcript.
Do not add assumptions or external facts.

Provide the output in this format:

Summary:
- ...

Action Items:
- ...

Decisions:
- ...

Transcript:
{transcript}
"""

    result = summarizer(
        prompt,
        max_length=300,
        min_length=100,
        do_sample=False
    )

    return result[0]["summary_text"]

st.title("📝 Meeting Transcript Summarizer")

st.markdown(
    """
This app summarizes meeting transcripts and extracts key action items and decisions.

⚠️ **Disclaimer:** This summary is AI-generated and may miss context.
Please verify important details.
"""
)

col1, col2 = st.columns(2)

with col1:
    transcript = st.text_area(
        "Meeting Transcript",
        height=350,
        placeholder="Paste meeting transcript here..."
    )

with col2:
    output = st.text_area(
        "AI-Generated Summary",
        height=350
    )

if st.button("Summarize"):
    if transcript:
        with st.spinner("Summarizing..."):
            output = summarize_meeting(transcript)
            st.session_state["output"] = output
    else:
        st.warning("Please paste a meeting transcript.")

if "output" in st.session_state:
    col2.text_area(
        "AI-Generated Summary",
        value=st.session_state["output"],
        height=350
    )
