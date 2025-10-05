"""
Tạo embs để tránh việc tính toán lại embs nhiều lần
"""

import tensorflow as tf
import pandas as pd
import numpy as np



# --- config path ---
ITEM_TOWER_PATH = "models/item_tower"
PRODUCTS_CSV_PATH = "data/products.csv"
OUTPUT_EMBEDDINGS_PATH = "data/prod_embs.npy"
OUTPUT_IDS_PATH = "data/prod_ids.npy"

def prepare_data(df):
    """Chuẩn bị dữ liệu đầu vào cho item_tower từ DataFrame."""
    
    return {
        "product_id": tf.constant([str(df["product_id"])]),
        "category": tf.constant([str(df["category"])]),
        "brand": tf.constant([str(df["brand"])]),
        "price": tf.constant([float(df["price"])], dtype=tf.float32),
        "description": tf.constant([str(df["description"])])
    }

def main():
    print("Tạo embeddings cho tất cả sản phẩm...")

    try:
        item_tower = tf.saved_model.load(ITEM_TOWER_PATH)
        print("load Item Tower success.")
    except Exception as e:
        print(f"Error load Item Tower: {e}")
        return

    try:
        products_df = pd.read_csv(PRODUCTS_CSV_PATH)
        products_df.fillna('', inplace=True)
        print(f"Succesful read csv {len(products_df)} sản phẩm từ CSV.")
    except Exception as e:
        print(f"Error read CSV: {e}")
        return

    all_embeddings = []
    all_product_ids = []

    
    print("\n Embed for each product")
    for index, row in products_df.iterrows():
        try:
            input_data = prepare_data(row)
            embedding_tensor = item_tower(input_data)
            all_embeddings.append(embedding_tensor.numpy().flatten())
            all_product_ids.append(str(row["product_id"]))
        except Exception as e:
            print(f"prod_id err {row['product_id']}: {e}")
            continue 

    
    embeddings_matrix = np.array(all_embeddings)
    ids_array = np.array(all_product_ids)
    print(f"\nEmbed for {len(embeddings_matrix)} products.")

    # save embs and ids to .npy
    np.save(OUTPUT_EMBEDDINGS_PATH, embeddings_matrix)
    np.save(OUTPUT_IDS_PATH, ids_array)
    print(f"Embeddings saved at: {OUTPUT_EMBEDDINGS_PATH}")
    print(f"Product IDs saved at: {OUTPUT_IDS_PATH}")

if __name__ == "__main__":
    main()