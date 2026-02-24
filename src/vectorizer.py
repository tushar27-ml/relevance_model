from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
import os
import pandas as pd


# ---------------------------------
# Vectorizer Class
# ---------------------------------

class HybridTfidfVectorizer:
    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=5000
        )

    def fit(self, texts):
        self.word_vectorizer.fit(texts)
        self.char_vectorizer.fit(texts)

    def transform(self, texts):
        word_features = self.word_vectorizer.transform(texts)
        char_features = self.char_vectorizer.transform(texts)
        return hstack([word_features, char_features])

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


# ---------------------------------
# Build + Save Function (OUTSIDE class)
# ---------------------------------

def build_and_save_vectorizer(dish_path, output_path):
    df = pd.read_csv(dish_path)

    vectorizer = HybridTfidfVectorizer()
    vectorizer.fit(df["dish_text"].tolist())

    joblib.dump(vectorizer, output_path)

    print(f"Vectorizer saved to {output_path}")
    print("Word vocab size:", len(vectorizer.word_vectorizer.vocabulary_))
    print("Char vocab size:", len(vectorizer.char_vectorizer.vocabulary_))


# ---------------------------------
# Script Entry Point
# ---------------------------------

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dish_path = os.path.join(BASE_DIR, "data/processed/dishes_processed.csv")
    output_path = os.path.join(BASE_DIR, "models/hybrid_vectorizer.pkl")

    build_and_save_vectorizer(dish_path, output_path)