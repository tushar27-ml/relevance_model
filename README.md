# Tiny Hybrid Semantic Retrieval + ML Ranker

## Problem Statement

At scale (e.g., Swiggy), user queries like:

> "milk sweet near me"

must be matched against hundreds of items in <20ms (GPU) or <100ms (CPU).

LLM APIs are too slow and expensive.

We need a:
- Tiny model (<10M parameters)
- Fast CPU inference
- No pretrained weights
- No HuggingFace models

---

## Solution Overview

We implement a **Hybrid Semantic Retrieval + Classical ML Ranker**.

### Architecture

Query
↓
TF-IDF similarity (fast lexical retrieval)
↓
Structured feature engineering
↓
Tiny MLP re-ranker
↓
Top-K ranked dishes

---

## Why Hybrid?

- Pure TF-IDF → fast but shallow
- Deep Transformers → slow and heavy
- Hybrid approach → fast + learnable

This mirrors production search stacks.

---

## Model Details

### Feature Vector (8 features)

1. TF-IDF cosine similarity
2. Word overlap ratio
3. Ingredient overlap ratio
4. Flavor match (binary)
5. Course match (binary)
6. Region match (binary)
7. Query length
8. Dish name length

### Ranker

- 2-layer MLP
- Architecture: 8 → 32 → 16 → 1
- Parameters: ~1.2K (<< 10M constraint)
- Loss: BCEWithLogitsLoss
- Query-level train/val split

---

### Inference Speed

Tested on CPU (standard laptop):

- 255 dishes → ~31ms
- Projected 500 dishes → ~70-80ms

✅ Meets <100ms CPU requirement

Batch scoring is fully vectorized.

---

## Model Size

- Ranker MLP: ~30 KB
- TF-IDF vectorizer: ~2–3 MB
- Scaler: negligible

Total model artifacts: **<5 MB**

✅ Meets <20MB constraint

---

## Multilingual Handling

Supports:
- English
- Hindi (transliterated)
- Hinglish

Example queries:
- "meetha jalebi"
- "milk sweet"
- "rasgulla from west bengal"
- "paneer wali dish"

Character n-gram TF-IDF enables robust matching for:
- Misspellings
- Transliteration
- Partial tokens

---

## Qualitative Examples

### Query: milk sweet
Top results:
- rabri
- rasabali
- lyangcha
- halvasan
- bebinca

### Query: milk kheer
Top results:
- kheer
- chak hao kheer
- rabri
- misti doi
- basundi

### Query: west bengal sweet
Top results:
- rasgulla
- sandesh
- misti doi
- ledikeni
- pantua

### Query: sweet jalebi
Top results:
- jalebi
- imarti
- chhena jalebi
- malpua
- gulab jamun

### Query: meetha laddu
Top results:
- laddu
- churma ladoo
- motichoor laddu
- besan laddu
- boondi laddu

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt