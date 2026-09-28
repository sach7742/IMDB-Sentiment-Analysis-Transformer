# 🎬 SentimentScope: IMDB Sentiment Analysis using Transformers

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)

An end-to-end PyTorch implementation of a Transformer architecture built from scratch for binary sentiment classification on the IMDB Movie Reviews dataset.

## 📌 Project Overview
This project adapts a Transformer block architecture (`DemoGPT`) for sequence classification instead of generative tasks. Using **subword tokenization (`bert-base-uncased`)**, **causal multi-head attention**, and **mean sequence pooling**, the model classifies movie reviews into **Positive (1)** or **Negative (0)** sentiment.

### Key Highlights
- **Custom Dataset & DataLoader Pipeline:** Built `IMDBDataset` using PyTorch `Dataset` with subword tokenization, dynamic padding, and sequence truncation ($T=128$).
- **Custom Transformer Architecture:** Modular implementation of `AttentionHead`, `MultiHeadAttention`, `FeedForward`, and `Block` modules with layer normalization and dropout.
- **Classification Projection Head:** Implemented sequence-level **mean pooling** to compress sequence representations into $d_{embed}=128$ vectors before linear projection into binary logits.
- **Evaluation & Performance:** Optimized using AdamW ($lr=3\times10^{-4}$) and Cross-Entropy Loss, achieving **>75% test accuracy** over 3 training epochs.
