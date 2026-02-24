import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from tqdm import tqdm
from vectorizer import HybridTfidfVectorizer

def build_hard_negative_dataset(
    dishes_path,
    queries_path,
    vectorizer_path,
    output_path,
    top_k=6,
    random_negatives=1
):
    """
    For each positive (query, dish):
    - Get TFIDF similarity against all dishes
    - Select top_k most similar (excluding true dish)
    - Add random_negatives random dishes
    """

    dishes = pd.read_csv(dishes_path)
    queries = pd.read_csv(queries_path)

    vectorizer = joblib.load(vectorizer_path)

    dish_texts = dishes["dish_text"].tolist()
    dish_names = dishes["name"].tolist()

    print("Computing dish embeddings...")
    dish_matrix = vectorizer.transform(dish_texts)

    rows = []

    for _, row in tqdm(queries.iterrows(), total=len(queries)):

        query = row["query"]
        true_dish = row["dish_name"]

        query_vec = vectorizer.transform([query])

        sims = cosine_similarity(query_vec, dish_matrix)[0]

        # Sort descending
        sorted_idx = np.argsort(-sims)

        # Add positive
        rows.append([query, true_dish, 1])

        # Collect hard negatives
        hard_count = 0
        for idx in sorted_idx:
            candidate = dish_names[idx]

            if candidate == true_dish:
                continue

            rows.append([query, candidate, 0])
            hard_count += 1

            if hard_count >= top_k:
                break

        # Add random negatives
        remaining = list(set(dish_names) - {true_dish})
        random_samples = np.random.choice(
            remaining,
            size=random_negatives,
            replace=False
        )

        for neg in random_samples:
            rows.append([query, neg, 0])

    df = pd.DataFrame(rows, columns=["query", "dish_name", "label"])

    print("\nDataset Stats:")
    print("Total samples:", len(df))
    print("Positives:", df["label"].sum())
    print("Negatives:", len(df) - df["label"].sum())

    df.to_csv(output_path, index=False)
    print("\nSaved to:", output_path)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    build_hard_negative_dataset(
        dishes_path=os.path.join(BASE_DIR, "data/processed/dishes_processed.csv"),
        queries_path=os.path.join(BASE_DIR, "data/processed/generated_queries_strict.csv"),
        vectorizer_path=os.path.join(BASE_DIR, "models/hybrid_vectorizer.pkl"),
        output_path=os.path.join(BASE_DIR, "data/processed/train_hard_negatives.csv"),
        top_k=6,
        random_negatives=1
    )