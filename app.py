# app.py
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import pandas as pd
import base64
from io import StringIO
import re
from groq import Groq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import openai

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"
if "symptom_history" not in st.session_state:
    st.session_state.symptom_history = []

# Set page config
st.set_page_config(page_title="Virtual Doctor Assistant", layout="wide")

# 💡 Sidebar UI
with st.sidebar:
    st.markdown("""
    <h2 style='color:#4CAF50'>🧠 Model Selection</h2>
    <p style='font-size: 14px;'>Choose the language model to generate medical responses:</p>
    """, unsafe_allow_html=True)

    static_groq_models = [
        "groq:deepseek-r1-distill-llama-70b",
        "groq:llama-3.1-8b-instant",
        "groq:llama-3.3-70b-versatile"
    ]
    gemini_models = [
        "gemini:gemini-1.5-pro",
        "gemini:gemini-2.0-flash",
                "gpt-4o"
    ]
    model_choice = st.radio("", static_groq_models + gemini_models, index=0)
    use_dataset = st.checkbox("📄 Use Dataset for Response", value=True)

# Medical Disclaimer
st.markdown("""
### 🏥 Virtual Doctor Assistant
<div style='border-left: 5px solid #f44336; padding: 10px;'>
<b>Disclaimer:</b> This tool is not a substitute for professional medical advice. Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

# Load the symptom description dataset
@st.cache_data
def load_symptom_data():
    path = "symptom_Description_enriched.csv"
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df

symptom_df = load_symptom_data()

# Extract keywords that match diseases
def extract_matching_rows(text, df):
    matches = []
    for _, row in df.iterrows():
        disease = row['Disease']
        pattern = re.compile(rf"\b{re.escape(disease.lower())}\b")
        if pattern.search(text.lower()):
            matches.append(row)
    return matches

# LLM setup
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_llm(messages, model):
    if model.startswith("groq:"):
        model = model.replace("groq:", "")
        response = client_groq.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content  # Groq response

    elif model.startswith("gemini:"):
        model = model.replace("gemini:", "")
        gemini_client = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))
        user_message = "".join([msg["content"] for msg in messages if msg["role"] == "user"])
        system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        full_prompt = f"{system_prompt}{user_message}"
        response = gemini_client.invoke(full_prompt)
        return response.content  # Gemini response

    elif model in ["gpt-4", "gpt-4o"]:
        from openai import AzureOpenAI
        client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=model
)
        chat_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        return chat_response.choices[0].message.content

    else:
        return "⚠️ Model selection error: Invalid model key. Please check your selection."


# Chat containers
chat_container = st.container()
input_container = st.container()

# Show history
with chat_container:
    for msg in st.session_state.chat_history.messages:
        if isinstance(msg, HumanMessage):
            st.markdown(f"""
            <div style='margin-top:10px; background-color:#1e1e1e; color:#e0e0e0; padding:10px; border-radius:10px; border:1px solid #444;'>
                <b>👤 You:</b> {msg.content}
            </div>
            """, unsafe_allow_html=True)
        elif isinstance(msg, SystemMessage):
            continue
        else:
            st.markdown(f"""
            <div style='margin-top:10px; background-color:#1e1e1e; color:#e0e0e0; padding:10px; border-radius:10px; border:1px solid #444;'>
                {msg.content}
            </div>
            """, unsafe_allow_html=True)

# Input box
with input_container:
    st.markdown("""
    <style>
        .stTextInput>div>div>input {
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #ccc;
        }
    </style>
    """, unsafe_allow_html=True)

    user_input = st.text_input("Describe your symptoms:", key="input", on_change=lambda: st.session_state.update({'send_trigger': True}))
    send_btn = st.button("🚀 Send")

# Handle user message
if (send_btn or st.session_state.get("send_trigger")) and user_input:
    st.session_state.chat_history.add_user_message(user_input)
    st.session_state.symptom_history.append(user_input)

    with st.spinner("Analyzing your input..."):
        try:
            matches = extract_matching_rows(user_input, symptom_df)
            highlighted_input = user_input
            html_blocks = []
            combined_text = ""

            if matches and use_dataset:
                for row in matches:
                    highlighted_input = re.sub(rf"\b{re.escape(row['Disease'].lower())}\b", f"**{row['Disease']}**", highlighted_input, flags=re.IGNORECASE)
                    html = f"""{row['Disease']}
Description: {row['Description']}
Precautions: {row['Precaution']}"""
                    html_blocks.append(html)
                    combined_text += f"Disease: {row['Disease']}\nDescription: {row['Description']}\nPrecaution: {row['Precaution']}\n\n"

                full_response = "<br>".join(html_blocks)
                for html in html_blocks:
                  st.session_state.chat_history.add_ai_message(html)
                st.markdown(html, unsafe_allow_html=True)
            else:
                combined_text = user_input

            full_symptom_context = "\n".join(st.session_state.symptom_history)

            if "deepseek" in model_choice:
                system_prompt = (
                    "You are a concise and helpful virtual doctor assistant. "
                    "Do not explain your thought process. Instead, respond briefly, directly, and kindly. "
                    "If symptoms are unclear, ask 1-2 simple follow-up questions. Avoid long reasoning."
                )
            else:
                system_prompt = """You are a friendly and supportive virtual doctor assistant.

Example:
User: I have fever and body pain.
Assistant: Thank you for sharing that. Can I ask:
- How long have you had the fever?
- Do you also feel chills, cough, or fatigue?

User: It started 2 days ago. I also feel tired.
Assistant (reasoning): Given the duration and fatigue, the symptoms could suggest flu.

🤔 Possible Condition: Influenza (Flu)
🧠 Reasoning: Fever, body pain, and tiredness are typical flu symptoms.
💡 Suggested Precautions: Rest, stay hydrated, and take paracetamol.
📊 Medicine: Paracetamol
🏥 Note: This is not medical advice.

Now, help the next user.

Instructions:
- If not enough context, ask 1-2 specific follow-up questions.
- If enough info, summarize condition and suggest safe steps.
- Think step-by-step before final answer."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation so far:\n{full_symptom_context}\n\nNew input:\n{user_input}"}
            ]

            response_text = query_llm(messages, model_choice)
            st.session_state.chat_history.add_ai_message(response_text)
            st.markdown(f"""
        <div style='margin-top:20px; background-color:#1e1e1e; color:#e0e0e0; padding:15px; border-radius:10px; border:1px solid #444; box-shadow: 0 0 10px rgba(0,0,0,0.15);'>
            <h4 style='color:#6FEAEA;'>🤖 {model_choice} Response</h4>
            <p>{response_text}</p>
        </div>
        """, unsafe_allow_html=True)

        except Exception as e:
            st.session_state.chat_history.add_ai_message(f"Error: {str(e)}")
            st.markdown(f"Error: {str(e)}")
    st.session_state.send_trigger = False
    st.rerun()

# Auto-scroll
st.markdown(r"""
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0, window.parent.document.querySelector('section.main').scrollHeight);
    </script>
""", unsafe_allow_html=True)
