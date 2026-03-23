# 🙄 Sarcasm Detector

A BERT-powered tweet sarcasm classifier built with Streamlit.

## Files in this repo

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System dependencies for Streamlit Cloud |

---

## Model Adding 

The `sarcasm_bert/` folder is uploaded to Hugging Face from Colab training and the end oint is added to the app.py file.


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

