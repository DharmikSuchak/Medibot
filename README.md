## 🩺 Medibot(Virtual Doctor Assistant)

**An Intelligent Multi-Model Conversational AI for Medical Symptom Analysis**

> ⚠️ *Disclaimer: This tool is not a substitute for professional medical advice. Always consult a qualified healthcare provider.*

---

### 📌 Overview

This project is a Streamlit-based conversational assistant designed to simulate a virtual doctor. Users can input their symptoms, and the assistant retrieves relevant disease information from a structured dataset and/or queries a selected LLM for further diagnosis suggestions.

It integrates:

* Multiple **state-of-the-art LLMs** (Groq-hosted and Gemini)
* A **RAG (Retrieval-Augmented Generation)** approach over a medical CSV dataset
* **Dynamic prompt engineering** and memory-preserving chat interface

---

### 🚀 Features

* Choose between **6+ LLMs** (Groq + Gemini)
* Integrated **medical CSV** with disease descriptions and precautions
* Intelligent response generation using **system prompts**
* Clean **chat-like UI** with scrolling and history
* **Keyword matching** + **semantic retrieval (optional)** via embeddings
* Supports **multi-turn conversation** with memory

---

### 📂 Project Structure

```
📦 Virtual-Doctor-Assistant
├── app.py                      # Main Streamlit application
├── symptom_Description_enriched.csv   # Cleaned and enriched dataset
├── requirements.txt            # Required Python dependencies
├── README.md                   # Project documentation (this file)
```

---

### 📊 Dataset Format

The assistant uses a custom CSV with the following columns:

| Disease  | Description                                | Precaution                        |
| -------- | ------------------------------------------ | --------------------------------- |
| Flu      | Viral infection with fever, fatigue, etc.  | Stay hydrated, rest, avoid cold   |
| Headache | Pain in head due to stress or other causes | Reduce screen time, hydrate, rest |

---

### 🤖 Supported LLMs

* **Groq API**:

  * `groq:deepseek-r1-distill-llama-70b`
  * `groq:gemma2-9b-it`
  * `groq:llama-3.1-8b-instant`
  * `groq:distil-whisper-large-v3-en`

* **Gemini API**:

  * `gemini:gemini-1.5-pro`
  * `gemini:gemini-2.0-flash`

---

### 🛠️ How to Run

1. **Clone the repo**:

```bash
git clone https://github.com/DharmikSuchak/Medibot.git
cd Medibot
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set API Keys** (Groq + Gemini):

```bash
export GROQ_API_KEY="your-groq-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

4. **Launch Streamlit app**:

```bash
streamlit run app.py
```

---

### 🧩 System Architecture
![Architecture Diagram](assets/flow.jpg)

### 📸 Screenshots
![UI Screenshot](assets/interface_demo.jpg)
---

### 🧪 Research Use Cases

This project supports:

* 🔬 Comparative evaluation of LLMs in healthcare
* 🧠 Prompt engineering for empathetic diagnosis
* 🧬 Hybrid RAG logic combining LLMs + structured data
* 📈 Accuracy, clarity, and safety analysis across models

---

### 🙌 Acknowledgments

* Groq and Google Gemini teams for LLM API access
* Streamlit for the front-end framework
* Open-source community for Sentence-BERT and FAISS

---
