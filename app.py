import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="AI Content Marketing Optimizer", layout="wide")

# ---------------------------
# Load the fine-tuned model
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "./AI_Content_Optimizer_Trained"
    device = 0 if torch.cuda.is_available() else -1

    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        device=device
    )
    return generator

generator = load_model()

# ---------------------------
# Build prompt
# ---------------------------
def build_prompt(platform, topic, tone, size):
    size_map = {
        "short": "1-2 sentences per idea.",
        "medium": "3-4 sentences per idea.",
        "long": "5-6 sentences per idea."
    }

    return (
        f"You are a social media content expert. "
        f"Generate {size_map[size]} for {platform} on '{topic}' "
        f"in a {tone} tone. Format the output as a numbered list."
    )

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🎯 AI Content Marketing Optimizer")
st.write("Generate marketing content using your fine-tuned LLM.")

col1, col2, col3 = st.columns(3)

with col1:
    platform = st.selectbox("Platform", ["Instagram", "YouTube", "TikTok", "Blog"])

with col2:
    tone = st.selectbox("Tone", ["friendly", "professional", "witty"])

with col3:
    size = st.selectbox("Content Size", ["short", "medium", "long"])

topic = st.text_input("Enter Topic (e.g., Fashion tips for college students)")

if st.button("Generate Content"):
    if topic.strip() == "":
        st.warning("Please enter a topic.")
    else:
        prompt = build_prompt(platform, topic, tone, size)

        with st.spinner("Generating content..."):
            output = generator(
                prompt,
                max_new_tokens=180,
                temperature=0.7,
                num_return_sequences=1
            )

        st.subheader("✨ AI-Generated Content")
        st.write(output[0]["generated_text"])
