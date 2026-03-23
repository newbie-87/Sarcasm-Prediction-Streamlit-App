# 🙄 Sarcasm Detector

A BERT-powered tweet sarcasm classifier built with Streamlit.

## Files in this repo

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System dependencies for Streamlit Cloud |
| `sarcasm_bert/` | Fine-tuned BERT model files (**you must add this**) |

---

## ⚠️ Important: Adding your model

The `sarcasm_bert/` folder is **not included** — you need to upload it from your Colab training run.

After training, run this in Colab to download the model folder:

```python
from google.colab import files
import shutil

shutil.make_archive("sarcasm_bert", "zip", ".", "sarcasm_bert")
files.download("sarcasm_bert.zip")
```

Then unzip it and place the `sarcasm_bert/` folder in the root of this repo before pushing to GitHub.

The folder should contain:
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

> **Note:** The free tier of Streamlit Cloud has 1 GB RAM. BERT fits within this but may take ~30 seconds to load on first visit.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
