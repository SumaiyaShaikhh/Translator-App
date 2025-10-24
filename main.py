# pip install streamlit openai-agents python-dotenv nest_asyncio

import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import os

# Fix event loop issues for Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found. Please add it to your .env or Streamlit secrets.")
    st.stop()

# Setup Gemini as OpenAI-compatible
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Translator agent
translator = Agent(
    name="Translator Agent",
    instructions="""
    You are a smart and accurate language translator.
    - Detect if the input is in English or Roman Urdu (Urdu written in Latin script).
    - If the input is in English, translate it into Roman Urdu.
    - If the input is in Roman Urdu, translate it into proper English.
    - You are only limited to translate between English and Roman Urdu — no other languages.
    - If the user inputs any language other than English or Roman Urdu, show an error message.
    - Only return the translation — no explanation.
    """
)

# --- Streamlit UI ---
st.set_page_config(page_title="Tranzify", layout="centered")

# Custom button styling
st.markdown(
    """
    <style>

    /* Glowing Translate Button */
    div.stButton > button:first-child {
        background-color: maroon;
        color: #FFFDE7;
        border: 2px solid #FFFACD;
        padding: 0.6em 1.3em;
        font-size: 18px;
        border-radius: 10px;
        font-weight: 800;
        letter-spacing: 0.5px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px #FFF9C4; /* light yellow glow */
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.07);
        box-shadow: 0 0 25px #FFFDE7;
        background-color: #4B0000; /* darker maroon */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown(
    """
    <h1 style='text-align:center; color:#DAA520; font-weight:900; font-family:Poppins, sans-serif;'>⚡ Tranzify</h1>
    <p style='text-align:center; color:maroon; font-size:21px; font-weight:800; font-family:Trebuchet MS, sans-serif;'>Your bilingual bridge — effortless English ⇄ Roman Urdu translation</p>
    """,
    unsafe_allow_html=True
)

# --- Input + Button (Enter key support) ---
with st.form(key="translate_form"):
    user_input = st.text_input("", placeholder="💬 Type something to translate...")
    translate_clicked = st.form_submit_button("TRANSLATE")

# --- Translation logic ---
if user_input and translate_clicked:
    with st.spinner("Translating... please wait."):
        try:
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(
                Runner.run(translator, input=user_input, run_config=config)
            )

            st.markdown(
                f"""
                <div style='background-color:#FFFBEA; padding:15px; border-radius:10px;
                font-size:18px; font-weight:700; color:maroon; font-family:Segoe UI, sans-serif;'>
                <b>Translation:</b><br>{response.final_output}
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
