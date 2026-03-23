# 🙄 Sarcasm Detector

A BERT-powered tweet sarcasm classifier built with Streamlit.

## Files in this repo

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System dependencies for Streamlit Cloud |

---

## ⚠️ Important: Adding your model

The `sarcasm_bert/` folder is uploaded to Hugging Face from Colab training .


The folder contains:
```
sarcasm_bert/
  config.json
  pytorch_model.bin   (or model.safetensors)
  tokenizer_config.json
  vocab.txt
  special_tokens_map.json
```

---

## Deploy on Streamlit Cloud

1. Push this repo (including `sarcasm_bert/`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo, branch (`main`), and set main file to `app.py`
5. Click **Deploy**

