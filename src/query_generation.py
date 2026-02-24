import pandas as pd
import ast

# ==========================
# HINGLISH MAP
# ==========================

HINGLISH_MAP = {
    "sweet": "meetha",
    "spicy": "teekha"
}

# ==========================
# STRICT QUERY GENERATION
# ==========================

def generate_queries_for_dish(row):
    name = row["name"]
    flavor = row["flavor_profile"]
    course = row["course"]
    state = row["state"]

    # ingredient_list stored as string, convert safely
    ingredients = row["ingredient_list"]
    if isinstance(ingredients, str):
        ingredients = ast.literal_eval(ingredients)

    queries = []

    # 1. Exact name
    queries.append(name)

    # 2. Name + course
    if course != "unknown":
        queries.append(f"{name} {course}")

    # 3. Flavor + name
    if flavor != "unknown":
        queries.append(f"{flavor} {name}")

    # 4. First ingredient + name
    if len(ingredients) > 0:
        main_ingredient = ingredients[0]
        queries.append(f"{main_ingredient} {name}")

    # 5. Region + name
    if state != "unknown":
        queries.append(f"{state} {name}")

    # 6. Hinglish flavor + name
    if flavor in HINGLISH_MAP:
        queries.append(f"{HINGLISH_MAP[flavor]} {name}")

    return list(set(queries))


def generate_all_queries(input_path, output_path):
    df = pd.read_csv(input_path)

    query_rows = []

    for _, row in df.iterrows():
        queries = generate_queries_for_dish(row)

        for q in queries:
            query_rows.append({
                "query": q,
                "dish_name": row["name"]
            })

    query_df = pd.DataFrame(query_rows)
    query_df.to_csv(output_path, index=False)

    print(f"Saved strict queries to {output_path}")
    print(f"Total queries generated: {len(query_df)}")


if __name__ == "__main__":
    generate_all_queries(
        input_path="../data/processed/dishes_processed.csv",
        output_path="../data/processed/generated_queries_strict.csv"
    )