import pandas as pd
import random

random.seed(42)

HARD_NEGATIVES = 6
RANDOM_NEGATIVES = 1


def get_hard_candidates(df, pos_row):
    """
    Hard candidates satisfy at least ONE:
    - Same course
    - Same flavor
    - Same region
    """

    hard_df = df[
        (
            (df["course"] == pos_row["course"]) |
            (df["flavor_profile"] == pos_row["flavor_profile"]) |
            (df["region"] == pos_row["region"])
        )
        &
        (df["name"] != pos_row["name"])
    ]

    return hard_df


def sample_negatives(df, positive_dish):
    pos_row = df[df["name"] == positive_dish].iloc[0]

    hard_candidates = get_hard_candidates(df, pos_row)

    # Sample structured hard negatives
    if len(hard_candidates) >= HARD_NEGATIVES:
        hard_samples = hard_candidates.sample(HARD_NEGATIVES)
    else:
        # If not enough hard candidates, take all
        hard_samples = hard_candidates

    hard_names = hard_samples["name"].tolist()

    # Semi-hard random negative
    remaining_df = df[
        (df["name"] != positive_dish) &
        (~df["name"].isin(hard_names))
    ]

    if len(remaining_df) > 0:
        random_sample = remaining_df.sample(RANDOM_NEGATIVES)
        random_names = random_sample["name"].tolist()
    else:
        random_names = []

    return hard_names + random_names


def build_training_dataset(dish_path, query_path, output_path):
    dishes = pd.read_csv(dish_path)
    queries = pd.read_csv(query_path)

    training_rows = []

    for _, row in queries.iterrows():
        query = row["query"]
        pos_dish = row["dish_name"]

        # Positive sample
        training_rows.append({
            "query": query,
            "dish_name": pos_dish,
            "label": 1
        })

        # Negatives
        neg_dishes = sample_negatives(dishes, pos_dish)

        for neg in neg_dishes:
            training_rows.append({
                "query": query,
                "dish_name": neg,
                "label": 0
            })

    train_df = pd.DataFrame(training_rows)
    train_df.to_csv(output_path, index=False)

    print(f"Training dataset saved to {output_path}")
    print(f"Total samples: {len(train_df)}")

    # Optional sanity check
    print("\nSanity Check:")
    print("Positives:", train_df["label"].sum())
    print("Negatives:", len(train_df) - train_df["label"].sum())


if __name__ == "__main__":
    build_training_dataset(
        dish_path="../data/processed/dishes_processed.csv",
        query_path="../data/processed/generated_queries_strict.csv",
        output_path="../data/processed/train_pairs_hard.csv"
    )