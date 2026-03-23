import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import re
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sarcasm Detector",
    page_icon="🙄",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0d0d0d;
    color: #f0ede6;
}

h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem !important;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #f0ede6 0%, #c8b99a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}

.subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.stTextArea textarea {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    color: #f0ede6 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 1rem !important;
    caret-color: #c8b99a;
}

.stTextArea textarea:focus {
    border-color: #c8b99a !important;
    box-shadow: 0 0 0 2px rgba(200, 185, 154, 0.15) !important;
}

.stButton > button {
    background: #f0ede6 !important;
    color: #0d0d0d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #c8b99a !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(200, 185, 154, 0.2) !important;
}

.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    border: 1px solid;
    animation: fadeIn 0.4s ease;
}

.result-sarcasm {
    background: rgba(255, 90, 90, 0.06);
    border-color: rgba(255, 90, 90, 0.25);
}

.result-regular {
    background: rgba(100, 220, 160, 0.06);
    border-color: rgba(100, 220, 160, 0.25);
}

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.6;
}

.result-verdict {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    line-height: 1;
    margin-bottom: 1rem;
}

.verdict-sarcasm { color: #ff6b6b; }
.verdict-regular { color: #6ddc9f; }

.confidence-bar-bg {
    background: #1e1e1e;
    border-radius: 999px;
    height: 6px;
    width: 100%;
    margin-top: 0.4rem;
    overflow: hidden;
}

.confidence-bar-fill-sarcasm {
    height: 100%;
    background: linear-gradient(90deg, #ff6b6b, #ff9d9d);
    border-radius: 999px;
    transition: width 0.6s ease;
}

.confidence-bar-fill-regular {
    height: 100%;
    background: linear-gradient(90deg, #6ddc9f, #a8f0c8);
    border-radius: 999px;
    transition: width 0.6s ease;
}

.conf-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #888;
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.25rem;
    margin-top: 0.8rem;
}

.example-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}

.chip {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    cursor: pointer;
    transition: all 0.2s;
}

.chip:hover {
    border-color: #c8b99a;
    color: #c8b99a;
}

.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 2rem 0;
}

.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #555;
    margin-bottom: 2rem;
}

.dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6ddc9f;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\bsarcasm\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Setting Hugging Face model repo ID here ──────────────────────────────────
HF_MODEL_ID = "newbie87/sarcasm-bert"
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_id: str = HF_MODEL_ID):
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model     = BertForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, max_len: int = 64):
    cleaned = clean_tweet(text)
    inputs  = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze()
    label = torch.argmax(probs).item()
    return label, probs[0].item(), probs[1].item()   # label, p_regular, p_sarcasm


# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading BERT model…"):
    tokenizer, model = load_model()


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("<h1>Sarcasm Detector</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">BERT · Tweet Classification</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="model-badge"><span class="dot"></span>bert-base-uncased · fine-tuned</div>',
    unsafe_allow_html=True
)

# Example tweets
EXAMPLES = [
    "Oh great, another Monday 🙄",
    "I absolutely love waiting in traffic for 2 hours.",
    "Just had the best coffee of my life.",
    "Yeah because that plan will definitely work out.",
    "The weather today is so beautiful!",
]

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
chosen_example = None
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    if col.button(f"#{i+1}", key=f"ex_{i}", help=ex):
        chosen_example = ex

# Text input
default_text = chosen_example if chosen_example else ""
tweet_input = st.text_area(
    "Enter a tweet",
    value=default_text,
    height=120,
    placeholder="Type or paste a tweet here…",
    label_visibility="collapsed"
)

analyse_btn = st.button("Analyse →")

# Result
if analyse_btn:
    if not tweet_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running inference…"):
            label, p_reg, p_sar = predict(tweet_input, tokenizer, model)

        is_sarcasm   = label == 1
        card_class   = "result-sarcasm"   if is_sarcasm else "result-regular"
        verdict_cls  = "verdict-sarcasm"  if is_sarcasm else "verdict-regular"
        verdict_text = "Sarcastic 🙄"     if is_sarcasm else "Genuine 😊"
        conf         = p_sar              if is_sarcasm else p_reg
        bar_class    = "confidence-bar-fill-sarcasm" if is_sarcasm else "confidence-bar-fill-regular"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label">Verdict</div>
            <div class="result-verdict {verdict_cls}">{verdict_text}</div>
            <div class="conf-label">
                <span>Confidence</span>
                <span>{conf*100:.1f}%</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="{bar_class}" style="width:{conf*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Regular probability**")
            st.progress(float(p_reg))
            st.markdown(
                f'<span style="font-family:Space Mono,monospace;font-size:0.8rem;color:#888">{p_reg*100:.2f}%</span>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown("**Sarcasm probability**")
            st.progress(float(p_sar))
            st.markdown(
                f'<span style="font-family:Space Mono,monospace;font-size:0.8rem;color:#888">{p_sar*100:.2f}%</span>',
                unsafe_allow_html=True
            )
