# DATA.md

## 1. Dataset Source

### Base Dataset
We use the **Indian Food 101** dataset (public Kaggle dataset):
- ~255 Indian dishes
- Attributes include:
  - name
  - ingredients
  - diet
  - flavor_profile
  - course
  - state
  - region

This dataset was selected because:
- It is publicly available
- It contains structured metadata
- It enables realistic query–dish relevance modeling
- It avoids proprietary or scraped data (per constraints)

No web scraping or API-based data collection was used.

---

## 2. Query Generation Strategy

Since real search logs were unavailable, we simulated realistic user queries using structured templates.

### 2.1 Strict Discriminative Queries

For each dish, we generated queries using combinations of:

- Dish name
- Ingredients
- Flavor profile
- Course
- Region
- State

Examples:
- `milk kheer`
- `sweet jalebi`
- `west bengal rasgulla`
- `maida flour dessert`
- `meetha gulab jamun`

These queries were designed to:
- Mimic realistic food search patterns
- Include attribute-based queries (region, flavor, ingredient)
- Avoid trivial duplication or noisy tokens

Total unique positive queries: **1443**

---

## 3. Hard Negative Mining

To simulate realistic retrieval noise, we implemented **semantic hard negative mining**.

### For each (query, true_dish):

1. Compute TF-IDF similarity between query and all dishes.
2. Select top-K similar dishes (excluding true dish).
3. Add:
   - 6 semantic hard negatives (highest similarity)
   - 1 random negative (for diversity)

This ensures negatives are:
- Structurally similar
- Lexically similar
- Often from same course/flavor category

Final dataset:

- Total samples: **11,544**
- Positives: **1,443**
- Negatives: **10,101**
- Ratio: 1:7 (positive:negative)

This better approximates real-world retrieval candidate sets.

---

## 4. Preprocessing

### Text Normalization
- Lowercasing
- Safe handling of missing values
- Parsing ingredient lists using `ast.literal_eval`
- Removal of NaN attributes

### Feature Engineering

For each query–dish pair, we compute:

1. TF-IDF cosine similarity (word + char n-grams)
2. Word overlap ratio
3. Ingredient overlap ratio
4. Flavor match (binary)
5. Course match (binary)
6. Region match (binary)
7. Query length
8. Dish name length

Exact name match feature was intentionally removed to prevent leakage.

---

## 5. Language Handling (Hindi / English / Hinglish)

The system supports multilingual inputs via:

- Word-level TF-IDF (1–2 grams)
- Character n-gram TF-IDF (3–5 grams)

Character n-grams enable:
- Robust handling of transliterated Hindi (e.g., "meetha", "paneer")
- Misspellings
- Hinglish queries
- Partial matches

No pretrained multilingual embeddings were used (per constraints).

---

## 6. Data Integrity & Leakage Prevention

We prevent leakage by:

- Performing train/validation split at the **query level**
- Removing exact name match feature
- Avoiding duplicated query–dish pairs
- Using semantic hard negatives instead of random negatives

Validation queries are strictly unseen during training.