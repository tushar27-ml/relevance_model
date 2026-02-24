import pandas as pd
import re
from typing import List


# ==========================
# TEXT NORMALIZATION
# ==========================

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_ingredients(ingredient_str: str) -> List[str]:
    if pd.isna(ingredient_str):
        return []
    ingredients = ingredient_str.split(",")
    return [normalize_text(i) for i in ingredients]


# ==========================
# MAIN PREPROCESSING
# ==========================

def preprocess_dataset(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    # Replace -1 with unknown
    df.replace("-1", "unknown", inplace=True)
    df.replace(-1, "unknown", inplace=True)

    # Normalize text columns
    text_columns = [
        "name",
        "diet",
        "flavor_profile",
        "course",
        "state",
        "region"
    ]

    for col in text_columns:
        df[col] = df[col].apply(normalize_text)

    # Process ingredients
    df["ingredient_list"] = df["ingredients"].apply(split_ingredients)

    # Flatten ingredients for text representation
    df["ingredients_text"] = df["ingredient_list"].apply(
        lambda x: " ".join(x)
    )

    # Build unified dish_text
    df["dish_text"] = (
        df["name"] + " "
        + df["ingredients_text"] + " "
        + df["diet"] + " "
        + df["flavor_profile"] + " "
        + df["course"] + " "
        + df["state"] + " "
        + df["region"]
    )

    df["dish_text"] = df["dish_text"].apply(normalize_text)

    # Keep relevant columns
    processed_df = df[
        [
            "name",
            "ingredient_list",
            "diet",
            "flavor_profile",
            "course",
            "state",
            "region",
            "dish_text"
        ]
    ]

    processed_df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to {output_path}")
    print(f"Total dishes: {len(processed_df)}")


if __name__ == "__main__":
    preprocess_dataset(
        input_path="../data/raw/indian_food.csv",
        output_path="../data/processed/dishes_processed.csv"
    )