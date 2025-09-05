# 📘 Machine Translation with Seq2Seq + Attention

## 🔹 Overview  
This project implements a **Neural Machine Translation (NMT)** system using a **Sequence-to-Sequence (Seq2Seq)** model with an **Attention mechanism**.  
It is trained on the **English ↔ German Multi30k dataset** (can be extended to other languages).  

Key features:  
- Encoder–Decoder architecture (LSTM)  
- Bahdanau-style Attention  
- BLEU score evaluation  
- Translation of custom sentences  
- Model saving and loading  

---

## 🔹 Requirements  
Run the project in **Google Colab** or locally with:  

```bash
pip install torch torchtext spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
