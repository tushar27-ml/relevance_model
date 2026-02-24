import time
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from feature_engineering import FeatureEngineer
from ranker import RankerMLP

from vectorizer import HybridTfidfVectorizer


class DishRanker:
    def __init__(
        self,
        dish_path,
        vectorizer_path,
        model_path,
        scaler_path
    ):
        self.dishes = pd.read_csv(dish_path)
        self.fe = FeatureEngineer(vectorizer_path)

        self.scaler = joblib.load(scaler_path)

        self.model = RankerMLP(input_dim=8)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # 🔥 Precompute dish TF-IDF matrix ONCE
        self.dish_texts = self.dishes["dish_text"].astype(str).tolist()
        self.dish_tfidf_matrix = self.fe.vectorizer.transform(self.dish_texts)

        print("Inference system loaded successfully.")
        print("Total dishes:", len(self.dishes))

    # ---------------------------------------
    # Fast Ranking
    # ---------------------------------------

    def rank(self, query, top_k=5):
        query = str(query).lower()

        start = time.time()

        # 1️⃣ Vectorize query ONCE
        query_vec = self.fe.vectorizer.transform([query])
        # 2️⃣ Cosine similarity against ALL dishes (vectorized)
        tfidf_similarities = cosine_similarity(
            query_vec,
            self.dish_tfidf_matrix
        ).flatten()

        # 3️⃣ Build structured features in vectorized way
        feature_matrix = []

        for idx, row in self.dishes.iterrows():

            ingredient_list = row["ingredient_list"]
            if isinstance(ingredient_list, str):
                try:
                    import ast
                    ingredient_list = ast.literal_eval(ingredient_list)
                except:
                    ingredient_list = []

            features = [
                tfidf_similarities[idx],
                self.fe.word_overlap(query, row["dish_text"]),
                self.fe.ingredient_overlap(query, ingredient_list),
                self.fe.contains_token(query, str(row["flavor_profile"])),
                self.fe.contains_token(query, str(row["course"])),
                self.fe.contains_token(query, str(row["region"])),
                len(query.split()),
                len(str(row["name"]).split())
            ]

            feature_matrix.append(features)

        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        
        print("Feature matrix shape:", feature_matrix.shape)
        # 4️⃣ Scale
        feature_matrix = self.scaler.transform(feature_matrix)

        # 5️⃣ Single batch forward pass
        with torch.no_grad():
            inputs = torch.tensor(feature_matrix, dtype=torch.float32)
            logits = self.model(inputs)
            scores = torch.sigmoid(logits).numpy().flatten()

        # 6️⃣ Rank
        results = list(zip(self.dishes["name"], scores))
        results.sort(key=lambda x: x[1], reverse=True)

        end = time.time()

        latency_ms = (end - start) * 1000
        print(len(self.fe.build_feature_vector("test", self.dishes.iloc[0])))
        return results[:top_k], latency_ms


if __name__ == "__main__":

    ranker = DishRanker(
        dish_path="../data/processed/dishes_processed.csv",
        vectorizer_path="../models/hybrid_vectorizer.pkl",
        model_path="../models/ranker_mlp.pt",
        scaler_path="../models/feature_scaler.pkl"
    )

    query = "west bengal milk sweet"
    
    results, latency = ranker.rank(query, top_k=5)

    print("\nQuery:", query)
    for dish,_ in results:
        print(f"{dish:30s}")

    print(f"\nLatency: {latency:.2f} ms")