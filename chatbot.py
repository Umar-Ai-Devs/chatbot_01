import json
import time
import streamlit as st
from dotenv import dotenv_values
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

import streamlit as st
from dotenv import dotenv_values

# Live environment: Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

# Local fallback for development
if not GROQ_API_KEY:
    config = dotenv_values(".env")
    GROQ_API_KEY = config.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY. Add it in Streamlit Secrets (live) or .env file (local).")
    st.stop()


# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="💬 GroqWise", page_icon="💬", layout="wide")

# ----------------------------
# Header with Branding
# ----------------------------
st.markdown(
    """
    <div style='text-align:center; margin-bottom:20px;'>
        <h1 style='margin:0; color:#4B7BEC; font-family:Montserrat, sans-serif; font-weight:700;'>💬 GroqWise</h1>
        <p style='margin:0; color:#777777; font-family:Open Sans, sans-serif; font-size:16px;'>
            by Umar_AI_Devs
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar.expander("Settings"):
    st.subheader("Controls")
    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.6)
    max_tokens = st.slider("Max Tokens", 50, 120, 80)

    # System prompt templates
    system_templates = {
        "Teaching Assistant": "You are a helpful concise teaching assistant. Use short, clear explanations suitable for demo.",
        "Friendly Chat": "You are a friendly and concise assistant. Keep responses short and demo-ready.",
        "Technical Support": "You are a technical support assistant. Provide step-by-step guidance concisely."
    }
    template_choice = st.selectbox("System Prompt Template", list(system_templates.keys()))
    system_prompt = st.text_area("Custom System Prompt", value=system_templates[template_choice])

    # Clear Chat
    if st.button("🧹 Clear Chat"):
        st.session_state.pop("history", None)
        st.session_state.pop("memory", None)
        st.session_state.memory = ConversationBufferMemory(return_messages=True, max_length=10)
        st.rerun()

# ----------------------------
# Memory & History Setup
# ----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, max_length=10)

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Predefined Demo Responses
# ----------------------------
demo_responses = {
    "How’s the weather today?": "It's a lovely day! Mostly sunny, 72°F high, 52°F low, gentle NW breeze.",
    "Explain Python lists": "Python lists are containers that hold multiple items. You can add, remove, or change items easily.",
    "Tell me a joke": "Why did the computer go to the doctor? Because it caught a virus! 😄"
}

# ----------------------------
# LLM Setup
# ----------------------------
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=GROQ_API_KEY
    )
else:
    st.session_state.llm.temperature = temperature
    st.session_state.llm.max_tokens = max_tokens

if "conv" not in st.session_state:
    st.session_state.conv = ConversationChain(
        llm=st.session_state.llm,
        memory=st.session_state.memory,
        verbose=False
    )

conv = st.session_state.conv

# ----------------------------
# Chat UI
# ----------------------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(0.5)
            try:
                # Use predefined demo response if available
                if user_input in demo_responses:
                    response = demo_responses[user_input]
                else:
                    response = conv.run(user_input)
                # Optional: Keep only first 2 sentences
                response_sentences = response.split(". ")
                response = ". ".join(response_sentences[:2]) + "." if len(response_sentences) > 2 else response
            except Exception as e:
                response = f"⚠️ Error: {e}"
            st.markdown(response)

    st.session_state.history.append({"role": "assistant", "content": response})

# Trim history for download & performance
st.session_state.history = st.session_state.history[-20:]

# ----------------------------
# Bottom Panel: Centered Buttons
# ----------------------------
if st.session_state.history:
    chat_txt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.history])

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.download_button(
            label="⬇️ Download Chat (TXT)",
            data=chat_txt,
            file_name="chat_history.txt",
            mime="text/plain"
        )
    with col2:
        if st.button("📝 Summarize Chat"):
            with st.spinner("Summarizing chat..."):
                try:
                    summary = conv.run(f"Summarize the following conversation:\n{chat_txt}")
                except Exception as e:
                    summary = f"⚠️ Error while summarizing: {e}"
                st.info(summary)

# ----------------------------
# Custom CSS for polished chat
# ----------------------------
st.markdown("""
<style>
    .stChatMessage > div:first-child {
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 4px;
    }
    .stChatMessage.user div:first-child {
        background-color: #F0F0F0;
        text-align: right;
    }
    .stChatMessage.assistant div:first-child {
        background-color: #E6F0FF;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)
