import pandas as pd
import numpy as np
import joblib
import ast
from sklearn.metrics.pairwise import cosine_similarity
from vectorizer import HybridTfidfVectorizer


class FeatureEngineer:
    def __init__(self, vectorizer_path):
        self.vectorizer = joblib.load(vectorizer_path)

    # ---------------------------
    # Utility Normalization
    # ---------------------------

    @staticmethod
    def safe_str(value):
        if pd.isna(value):
            return ""
        return str(value).lower().strip()

    @staticmethod
    def safe_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except Exception:
                return []
        return []

    # ---------------------------
    # Core Feature Computations
    # ---------------------------

    def tfidf_similarity(self, query, dish_text):
        q_vec = self.vectorizer.transform([query])
        d_vec = self.vectorizer.transform([dish_text])
        return cosine_similarity(q_vec, d_vec)[0][0]

    def word_overlap(self, query, dish_text):
        q_tokens = set(query.split())
        d_tokens = set(dish_text.split())

        if not q_tokens:
            return 0.0

        return len(q_tokens & d_tokens) / len(q_tokens)

    def ingredient_overlap(self, query, ingredient_list):
        q_tokens = set(query.split())
        ingredients = set(ingredient_list)

        if not q_tokens:
            return 0.0

        return len(q_tokens & ingredients) / len(q_tokens)

    def contains_token(self, query, token):
        if not token:
            return 0
        return int(token in query)

    # ---------------------------
    # Build Feature Vector
    # ---------------------------

    def build_feature_vector(self, query, dish_row):
        query = self.safe_str(query)

        name = self.safe_str(dish_row["name"])
        dish_text = self.safe_str(dish_row["dish_text"])
        flavor = self.safe_str(dish_row["flavor_profile"])
        course = self.safe_str(dish_row["course"])
        region = self.safe_str(dish_row["region"])

        ingredient_list = self.safe_list(dish_row["ingredient_list"])

        features = []

        # 1. TF-IDF similarity
        features.append(self.tfidf_similarity(query, dish_text))

        # 2. Word overlap
        features.append(self.word_overlap(query, dish_text))

        # 3. Ingredient overlap
        features.append(self.ingredient_overlap(query, ingredient_list))

        # 4. Flavor match
        features.append(self.contains_token(query, flavor))

        # 5. Course match
        features.append(self.contains_token(query, course))

        # 6. Region match
        features.append(self.contains_token(query, region))

        # 7. Exact name match
        #features.append(int(query == name))

        # 8. Query length
        features.append(len(query.split()))

        # 9. Dish name length
        features.append(len(name.split()))

        return np.array(features, dtype=np.float32)


# ---------------------------------
# Dataset Feature Builder
# ---------------------------------

def build_feature_dataset(train_pairs_path, dish_path, vectorizer_path, output_path):
    train_df = pd.read_csv(train_pairs_path)
    dishes = pd.read_csv(dish_path)

    fe = FeatureEngineer(vectorizer_path)

    # Fast lookup
    dish_dict = {row["name"]: row for _, row in dishes.iterrows()}

    feature_rows = []

    for _, row in train_df.iterrows():
        query = row["query"]
        dish_name = row["dish_name"]
        label = row["label"]

        dish_row = dish_dict[dish_name]

        features = fe.build_feature_vector(query, dish_row)

        feature_rows.append(np.append(features, label))

    feature_array = np.array(feature_rows, dtype=np.float32)

    np.savez(
    output_path,
    X=feature_array[:, :-1],
    y=feature_array[:, -1],
    queries=train_df["query"].values
)

    print(f"Feature matrix saved to {output_path}")
    print("Shape:", feature_array.shape)
    print("Positive samples:", int(feature_array[:, -1].sum()))
    print("Negative samples:", len(feature_array) - int(feature_array[:, -1].sum()))


if __name__ == "__main__":
    build_feature_dataset(
        train_pairs_path="../data/processed/train_hard_negatives.csv",
        dish_path="../data/processed/dishes_processed.csv",
        vectorizer_path="../models/hybrid_vectorizer.pkl",
        output_path="../data/processed/features.npz"
    )