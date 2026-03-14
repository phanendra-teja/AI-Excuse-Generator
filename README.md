# 🎭 AI Excuse Generator — Intelligent Context-Aware Excuse & Proof Creation System

An AI-powered system that generates detailed, realistic, context-aware excuses accompanied by authentic-looking proof documents — including fake chat screenshots, official-style certificates, downloadable PDFs, and voice messages. Built using Google's **Gemma-2B-IT** large language model with a **Gradio** web interface.

Developed as the **AI/ML Capstone Project** during internship at **LaunchED**, NIT Puducherry (NITPY).

---

## 📌 Project Overview

| Detail | Value |
|---|---|
| Task | AI-powered excuse generation + multi-format proof creation |
| Model | `google/gemma-2b-it` (2B parameter instruction-tuned LLM) |
| Interface | Gradio web UI |
| Platform | Google Colab (GPU-accelerated) |
| Quantization | 4-bit (NF4) via `bitsandbytes` |
| Output Formats | Text, PDF, Chat Image (JPG), Certificate (PNG), Voice (MP3) |

---

## 🎯 Objectives

- Generate realistic, long-form, context-aware excuses using a state-of-the-art LLM
- Provide **4 types of proof** to support every generated excuse
- Design an intuitive Gradio UI with personalized inputs (name, institute, tone, urgency, reason, context)
- Run entirely on **Google Colab** — no local setup or high-end hardware needed

---

## 🧠 Model — Google Gemma-2B-IT

| Property | Details |
|---|---|
| Model ID | `google/gemma-2b-it` |
| Parameters | 2 Billion |
| Type | Instruction-tuned transformer |
| Quantization | 4-bit NF4 (`bitsandbytes`) |
| Why chosen | Lightweight yet fluent, open license, works on consumer-grade GPUs |

### Inference Configuration

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

output = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)
```

---

## 🏗️ System Architecture

```
User Input (Gradio UI)
        │
        ▼
Excuse Generator
(Gemma-2B-IT via Transformers)
        │
        ▼
┌───────────────────────────┐
│       Proof Generators     │
│  ┌─────────────────────┐  │
│  │   PDF Generator     │  │  ← FPDF letter format
│  ├─────────────────────┤  │
│  │ Chat Screenshot     │  │  ← PIL ImageDraw fake chat
│  ├─────────────────────┤  │
│  │ Certificate Creator │  │  ← PIL fake official cert
│  ├─────────────────────┤  │
│  │ Voice Message (TTS) │  │  ← gTTS MP3
│  └─────────────────────┘  │
└───────────────────────────┘
        │
        ▼
Return Outputs to User (Gradio UI)
```

---

## 🌟 Key Features

- **Real-time LLM excuse generation** — long, detailed, human-like output tailored to your context
- **4 types of downloadable proof** per session:
  - 📄 Formal excuse letter (PDF)
  - 💬 Fake chat screenshot mimicking realistic teacher/boss conversations (JPG)
  - 🏅 Fake official certificate resembling authority-issued leave approvals (PNG)
  - 🎙️ MP3 voice message narrating the excuse via TTS
- **Personalized output** — inputs include name, institute, context, reason, urgency, and tone
- **Clean UI** — Gradio dropdowns and textboxes with a single "Submit" click
- **Unique filenames** per session using `uuid` — no file conflicts

---

## 🖌️ User Interface

Built using Gradio, the UI collects the following inputs:

| Input | Type | Options |
|---|---|---|
| Your Name | Textbox | Free text |
| Institute / Company | Textbox | Free text |
| Context | Dropdown | Work, School, Social, Family |
| Reason | Dropdown | Health, Personal, Emergency, Technical |
| Urgency | Dropdown | Low, Medium, High |
| Tone | Dropdown | Professional, Emotional, Funny, Casual |

### Empty UI

![Empty UI](ui_empty.png)

### Filled Inputs

![Filled UI](ui_filled.png)

### Context Dropdown

![Context Dropdown](ui_dropdown.png)

---

## 📄 Outputs & Example

### Example Inputs

| Field | Value |
|---|---|
| Name | Phanendra Teja V |
| Institute | NIT Puducherry |
| Context | School |
| Reason | Health |
| Urgency | Medium |
| Tone | Professional |

### Generated Excuse + Downloadable Files

![Output UI with files](ui_output.png)

*The UI shows the generated excuse text on the right, with downloadable files: PDF, chat image, and certificate.*

---

### 📄 PDF Excuse Letter

Formal letter format addressed "To Whom It May Concern" — generated and downloaded as a `.pdf`.

![PDF Output](output_pdf.png)

---

### 💬 Fake Chat Screenshot

A realistic teacher–student chat exchange embedding the generated excuse naturally.

![Chat Screenshot](output_chat.png)

---

### 🏅 Fake Certificate

An official-style Medical Fitness Certificate with name, institute, context, reason, date, and issuing authority.

![Certificate](output_certificate.png)

---

## 🧩 Modules

| Module | Purpose |
|---|---|
| `transformers` | Load and run Gemma-2B-IT for text generation |
| `torch` | Tensor computation and GPU inference |
| `gradio` | Frontend web UI with file downloads |
| `fpdf` | Generate downloadable PDF excuse letters |
| `PIL / Pillow` | Create fake chat screenshots and certificates |
| `gtts` | Generate MP3 voice output from excuse text |
| `bitsandbytes` | 4-bit model quantization for memory efficiency |
| `uuid` | Unique filenames per output session |
| `textwrap` | Wrap long text for image rendering |
| `huggingface_hub` | Model access and HF token authentication |

---

## 🛠️ Tech Stack

| Layer | Tool / Technology |
|---|---|
| Language Model | `google/gemma-2b-it` (Hugging Face) |
| Framework | Python, Transformers |
| UI | Gradio |
| Platform | Google Colab |
| Media Generation | FPDF, Pillow (PIL), gTTS |
| File Management | uuid, os |

---

## 🚀 How to Run

### ▶️ Run on Google Colab (Recommended)

> 👉 **[Open in Google Colab](https://colab.research.google.com/drive/1XL4MT8s6G9jlTgGdnIl1OyaFia5-LHSv?usp=sharing)**

⚠️ **Enable GPU**: Runtime → Change runtime type → **T4 GPU**

### Step-by-Step

**1. Install dependencies**
```bash
!pip install transformers accelerate bitsandbytes gradio fpdf pillow gtts
```

**2. Authenticate Hugging Face**
```python
from huggingface_hub import login
login("your_hf_token_here")
```
> Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).  
> Request access to `google/gemma-2b-it` at [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it).

**3. Load model with 4-bit quantization**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)
```

**4. Launch Gradio UI**
```python
import gradio as gr
gr.Interface(fn=full_generator, ...).launch()
```

---

## 📁 Project Structure

```
AI-Excuse-Generator/
│
├── MajorProject_ExcuseGenerator.ipynb    # Main Colab notebook
├── README.md                             # This file
│
# Screenshots
├── ui_empty.png                          # Empty Gradio UI
├── ui_filled.png                         # UI with example inputs
├── ui_output.png                         # UI showing generated output + files
├── ui_dropdown.png                       # Context dropdown options
├── output_pdf.png                        # Generated PDF excuse letter
├── output_chat.png                       # Generated fake chat screenshot
├── output_certificate.png               # Generated fake certificate
│
# Generated at runtime (not committed):
├── excuse_<uuid>.pdf
├── chat_<uuid>.jpg
├── cert_<uuid>.png
└── voice_<uuid>.mp3
```

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| Gated model access on Hugging Face | Requested model access and enabled HF token permissions |
| Generated excuse too short | Modified prompt structure and set `max_new_tokens=300+` |
| Colab GPU memory issues | Applied 4-bit NF4 quantization via `bitsandbytes` |
| Model trust error on load | Set `trust_remote_code=True` in `from_pretrained()` |
| PDF and image text overflow | Used `textwrap` wrapping and `ImageDraw` coordinate fixes |

---

## 🎯 Future Improvements

- Add **email integration** to auto-send excuses to professors/managers
- Expand to **multi-language voice output** (Hindi, Telugu, etc.)
- Add **calendar-aware excuse generation** (context from today's date and events)
- Allow users to **write their own excuse text** and only generate proof documents
- Add **user login and history tracking** for saved excuse sessions
- Deploy as a **Streamlit or Flask web app** for permanent public access

---

## 👤 Author

**Phanendra Teja V**  
B.Tech CSE — NIT Puducherry (NITPY), Batch 2024–2028  
AI/ML Internship Capstone — LaunchED

---

## 🔗 References

- [Gemma-2B Model Card — Hugging Face](https://huggingface.co/google/gemma-2b-it)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [Gradio Documentation](https://www.gradio.app/docs)
- [FPDF for Python](https://pyfpdf.readthedocs.io/)
- [gTTS — Google Text-to-Speech](https://pypi.org/project/gTTS/)
- [Pillow (PIL) Documentation](https://pillow.readthedocs.io/)

---

## 📄 License

This project is for educational purposes — developed as an AI/ML internship capstone.
