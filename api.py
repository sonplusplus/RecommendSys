"""
api và hybrid model (kết hợp collaborative(lọc cộng tác) và content_based(lọc nội dung sản phẩm))
content_based được lấy từ item_tower
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn

# --- Config ---
COLLABORATIVE_MODEL_PATH = "models/rs_model"
ITEM_TOWER_PATH = "models/item_tower"
PRODUCTS_CSV_PATH = "data/products.csv"
EMBEDDINGS_PATH = "data/prod_embs.npy"
IDS_PATH = "data/prod_ids.npy"

app = FastAPI(
    title="Recommendation System API",
    description="API cho hệ thống gợi ý sản phẩm của Tech Shop",
    version="1.0.0"
)

# Load models and data
try:
    print("load models")
    #load collaborative 
    collab = tf.saved_model.load(COLLABORATIVE_MODEL_PATH)
    
    #load item_tower
    item_tower = tf.saved_model.load(ITEM_TOWER_PATH)
    
    #load embs from prep_emb.py avoid recompute
    all_product_embeddings = np.load(EMBEDDINGS_PATH)
    all_product_ids = np.load(IDS_PATH).tolist()
    
    #load prod data
    products_df = pd.read_csv(PRODUCTS_CSV_PATH).astype({'product_id': 'str'})
    products_df.set_index('product_id', inplace=True)
    products_df.fillna('', inplace=True)

    print("Tải thành công!")
except Exception as e:
    print(f"Lỗi : Không thể tải model hoặc dữ liệu. API sẽ không hoạt động. Lỗi: {e}")
    collab = item_tower = None # Vô hiệu hóa model nếu có lỗi


#
def get_product_features_as_tensor(product_id: str):
    try:
        product_info = products_df.loc[product_id]
        return {
            "product_id": tf.constant([str(product_id)]),
            "category": tf.constant([str(product_info["category"])]),
            "brand": tf.constant([str(product_info["brand"])]),
            "price": tf.constant([float(product_info["price"])], dtype=tf.float32),
            "description": tf.constant([str(product_info["description"])])
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No prod_id '{product_id}' match found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# API Endpoints
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Chào mừng đến với hệ thống gợi ý!"}

@app.get("/recommend/content-based/{product_id}", response_model=List[str], tags=["Recommendation"])
def get_content_based_recommendations(product_id: str, top_k: int = 10):
    """
    Item-to-Item
    dựa trên nội dung sản phẩm
    """
    if not item_tower:
        raise HTTPException(status_code=503, detail="Dịch vụ gợi ý chưa sẵn sàng.")
    
    #emb của sp target 
    target_features = get_product_features_as_tensor(product_id)
    target_embedding = item_tower(target_features).numpy()

    #cosine similarity, hệ số góc giữa 2 vector prod
    #tích vô hướng/tích độ dài
    similarities = np.dot(all_product_embeddings, target_embedding.T).flatten() #vô hướng
    
    # Chuẩn hóa (normalize)
    norms = np.linalg.norm(all_product_embeddings, axis=1) * np.linalg.norm(target_embedding)
    
    similarities = similarities / norms

    
    #top_k sp tương đồng,skip sản phẩm 1 vì nó chính là sp target
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    similar_product_ids = [all_product_ids[i] for i in similar_indices]
    return similar_product_ids

@app.get("/recommend/hybrid/{user_id}/{product_id}", response_model=List[str], tags=["Recommendation"])
def get_hybrid_recommendations(user_id: str, product_id: str, top_k: int = 10):
    """
    hybrid model
    (40%) kết hợp lọc cộng tác (user-to-item)
    (60%) và lọc nội dung (item-to-item)
    """
    if not collab:
        raise HTTPException(status_code=503, detail="Dịch vụ gợi ý chưa sẵn sàng.")
    
    num_collaborative = int(top_k * 0.4)
    num_content = top_k - num_collaborative 

    # collab
    try:
        _, collaborative_recs_tensor = collab(tf.constant([user_id]))
        collaborative_recs = [item.decode('utf-8') for item in collaborative_recs_tensor[0, :num_collaborative].numpy()]
    except Exception:
        collaborative_recs = []
    
    # content
    content_recs = get_content_based_recommendations(product_id, top_k=num_content)

    # hybrid
    final_recs = []
    seen_ids = set()

    for pid in collaborative_recs:
        if pid not in seen_ids:
            final_recs.append(pid)
            seen_ids.add(pid)
            
    for pid in content_recs:
        if pid not in seen_ids and len(final_recs) < top_k:
            final_recs.append(pid)
            seen_ids.add(pid)
    
    if not final_recs:
        return content_recs

    return final_recs

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)