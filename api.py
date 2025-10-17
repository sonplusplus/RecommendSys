import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException, Query
from typing import Dict, Optional
import uvicorn

from als import ALSModel


ALS_MODEL_PATH = "models/als_model.pkl"
PHOBERT_EMBEDDINGS_PATH = "data/phobert_embeddings.pkl"
PRODUCTS_CSV_PATH = "data/products.csv"

app = FastAPI(
    title="Hybrid Recommendation System",
    description="Weighted ALS Collaborative Filtering + PhoBERT Content-Based",
)

als_model = None
all_product_embeddings = None
all_product_ids = None
products_df = None

#LOAD MODELS

try:    
    with open(ALS_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    als_model = ALSModel(model_data)
    

    print(f"  Users: {len(als_model.user_map):,}")
    print(f"  Items: {len(als_model.item_map):,}")
    print(f"  Latent factors: {als_model.n_factors}")
    print(f"  Alpha (confidence): {als_model.alpha}")
    print(f"  User personalization: {'Enabled' if als_model.has_user_personalization() else 'Disabled'}")
    
except FileNotFoundError:
    print("ALS(collab) model not found at", ALS_MODEL_PATH)
except Exception as e:
    print(f"Error loading ALS model: {e}")

try:
    print("LOADING PHOBERT EMBEDDINGS")
    print("="*70)
    
    with open(PHOBERT_EMBEDDINGS_PATH, 'rb') as f:
        phobert_data = pickle.load(f)
    
    all_product_embeddings = phobert_data['embeddings']
    all_product_ids = phobert_data['product_ids']
    
    print("PhoBERT embeddings loaded successfully")
    print(f"  Model: {phobert_data['model_name']}")
    print(f"  Products: {len(all_product_ids):,}")
    print(f"  Embedding dim: {all_product_embeddings.shape[1]}")
    print(f"  Normalized: {phobert_data.get('normalized', False)}")
    
except FileNotFoundError:
    print("PhoBERT embs not found at", PHOBERT_EMBEDDINGS_PATH)
except Exception as e:
    print(f"Error loading PhoBERT: {e}")

try:
    print("\n" + "="*70)
    print("LOADING PRODUCT METADATA")
    print("="*70)
    
    products_df = pd.read_csv(PRODUCTS_CSV_PATH)
    products_df['product_id'] = products_df['product_id'].astype(str)
    products_df.set_index('product_id', inplace=True)
    products_df.fillna('', inplace=True)
    
    print(f"✓ Product data loaded: {len(products_df):,} products")
    
except FileNotFoundError:
    print("products ds not found at", PRODUCTS_CSV_PATH)
except Exception as e:
    print(f"Error loading products: {e}")

#final status

if als_model and all_product_embeddings is not None and products_df is not None:
    print("all model load success")
else:
    print("1 or 2 model fail to load")




def get_product_info(product_id: str) -> Dict:
    """Get product metadata"""
    if products_df is None:
        raise HTTPException(status_code=503, detail="Product data not available")
    
    try:
        info = products_df.loc[product_id]
        return {
            'product_id': product_id,
            'name': info.get('name', info.get('product_name', '')),
            'category': info.get('category', ''),
            'brand': info.get('brand', ''),
            'price': float(info.get('price', 0)),
            'description': str(info.get('description', ''))[:150] + '...' if info.get('description') else ''
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")


def get_popular_products(top_k: int = 10):
    if products_df is None:
        return []
    
    try:
        if 'popularity_score' in products_df.columns:
            popular = products_df.nlargest(top_k, 'popularity_score')
        elif 'views' in products_df.columns:
            popular = products_df.nlargest(top_k, 'views')
        else:
            popular = products_df.sample(min(top_k, len(products_df)))
        
        return popular.index.tolist()
    except:
        return products_df.index[:top_k].tolist()


#api endpoints

@app.get("/", tags=["General"])
def root():
    """Health check and API info"""
    return {
        "service": "Hybrid Recommendation System",
        "version": "4.0-weighted",
        "status": "online" if (als_model and all_product_embeddings is not None) else "degraded",
        "features": {
            "user_personalization": als_model is not None and als_model.has_user_personalization(),
            "item_similarity": als_model is not None,
            "content_based": all_product_embeddings is not None,
            "weighted_interactions": True  # NEW
        },
        "models": {
            "collaborative": {
                "type": "Weighted ALS (Matrix Factorization)",
                "status": "ready" if als_model else "unavailable",
                "users": len(als_model.user_map) if als_model else 0,
                "items": len(als_model.item_map) if als_model else 0,
                "supports_weights": True,  # NEW
                "alpha": als_model.alpha if als_model else None  # NEW
            },
            "content_based": {
                "type": "PhoBERT (Vietnamese NLP)",
                "status": "ready" if all_product_embeddings is not None else "unavailable",
                "products": len(all_product_ids) if all_product_ids else 0
            }
        },
        "endpoints": {
            "homepage": "/recommend/homepage",
            "product_detail": "/recommend/product-detail/{product_id}",
            "content_only": "/recommend/content-based/{product_id}",
            "collaborative_only": "/recommend/similar-items/{product_id}"
        },
        "interaction_weights": {
            "click": 1,
            "add_to_cart": 2,
            "purchase": 3
        }
    }


@app.get("/recommend/homepage", tags=["Recommendations"])
def homepage_recommendations(
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    if user_id and als_model and als_model.has_user_personalization():
        if user_id in als_model.user_map:
            try:

                recs_with_scores = als_model.predict_for_user_with_scores(user_id, top_k=top_k)
                
                if recs_with_scores:
                    detailed_recs = []
                    for rec in recs_with_scores:
                        try:
                            product_info = get_product_info(rec['product_id'])
                            product_info['confidence'] = rec['confidence']
                            product_info['raw_score'] = rec['score']
                            detailed_recs.append(product_info)
                        except:
                            continue
                    
                    return {
                        "user_id": user_id,
                        "mode": "personalized",
                        "method": "weighted_collaborative_filtering",
                        "title": "Gợi ý cho bạn",
                        "description": "Dựa trên lịch sử tương tác của bạn (xem, thêm giỏ, mua)",
                        "total_recommendations": len(detailed_recs),
                        "recommendations": detailed_recs
                    }
            except Exception as e:
                print(f"Error in personalization: {e}")
    
    # Fallback to popular products
    print(f"Fallback to popular products (user_id={user_id})")
    popular_ids = get_popular_products(top_k=top_k)
    
    detailed_recs = []
    for rec_id in popular_ids:
        try:
            product_info = get_product_info(rec_id)
            detailed_recs.append(product_info)
        except:
            continue
    
    return {
        "user_id": user_id,
        "mode": "popular" if not user_id else "cold_start",
        "method": "popularity_based",
        "title": "Sản phẩm phổ biến" if not user_id else "Khám phá sản phẩm",
        "description": "Sản phẩm được nhiều người quan tâm",
        "total_recommendations": len(detailed_recs),
        "recommendations": detailed_recs
    }


@app.get("/recommend/content-based/{product_id}", tags=["Recommendations"])
def content_based_recommendations(product_id: str, top_k: int = 10):    #phoBERT

    if all_product_embeddings is None:
        raise HTTPException(status_code=503, detail="Content model not available")
    
    if product_id not in all_product_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in embeddings"
        )
    
    try:
        prod_idx = all_product_ids.index(product_id)
        target_embedding = all_product_embeddings[prod_idx]
        
        similarities = np.dot(all_product_embeddings, target_embedding)      
 
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        recommendations = []
        for i in top_indices:
            try:
                product_info = get_product_info(all_product_ids[i])
                product_info['similarity'] = float(similarities[i])
                recommendations.append(product_info)
            except:
                continue
        
        return {
            "product_id": product_id,
            "method": "content_based",
            "algorithm": "PhoBERT + Cosine Similarity",
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/recommend/similar-items/{product_id}", tags=["Recommendations"])
def similar_items_collaborative(product_id: str, top_k: int = 10):
    if not als_model:
        raise HTTPException(status_code=503, detail="ALS model not available")
    
    similar = als_model.get_similar_items(product_id, top_k=top_k)
    
    if similar is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in training data"
        )
    
    # Enrich with product details
    detailed_recs = []
    for sim_id in similar:
        try:
            product_info = get_product_info(sim_id)
            detailed_recs.append(product_info)
        except:
            continue
    
    return {
        "product_id": product_id,
        "method": "collaborative_similarity",
        "algorithm": "Weighted ALS Item Factors + Cosine Similarity",
        "description": "Sản phẩm mà người dùng thường tương tác cùng nhau",
        "total_recommendations": len(detailed_recs),
        "similar_items": detailed_recs
    }


@app.get("/recommend/product-detail/{product_id}", tags=["Recommendations"])
def product_detail_recommendations(
    product_id: str, 
    top_k: int = 10,
    user_id: Optional[str] = None
):
    
    if not als_model or all_product_embeddings is None:
        raise HTTPException(status_code=503, detail="Models not available")
    
    if product_id not in all_product_ids:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    
    if user_id and als_model.has_user_personalization() and user_id in als_model.user_map:
        STRATEGY = "logged_in"
        COLLAB_WEIGHT = 0.4
        CONTENT_WEIGHT = 0.6
        num_collab = int(top_k * COLLAB_WEIGHT)
        num_content = top_k - num_collab
    else:
        STRATEGY = "guest"
        COLLAB_WEIGHT = 0.0
        CONTENT_WEIGHT = 1.0
        num_collab = 0
        num_content = top_k
    
    # Get only collab
    collab_recs = []
    if num_collab > 0 and product_id in als_model.item_map:
        try:
            collab_recs = als_model.get_similar_items(product_id, top_k=num_collab) or []
        except:
            pass
    
    # Get only content-based 
    content_recs = []
    try:
        prod_idx = all_product_ids.index(product_id)
        target_embedding = all_product_embeddings[prod_idx]
        similarities = np.dot(all_product_embeddings, target_embedding)
        top_indices = np.argsort(similarities)[::-1][1:num_content+1]
        content_recs = [all_product_ids[i] for i in top_indices]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    # Merge recommendations 
    final_recs = []
    seen = set([product_id])
    
    for pid in collab_recs:
        if pid not in seen:
            final_recs.append(pid)
            seen.add(pid)
    
    for pid in content_recs:
        if pid not in seen and len(final_recs) < top_k:
            final_recs.append(pid)
            seen.add(pid)
    
    if len(final_recs) < top_k:
        prod_idx = all_product_ids.index(product_id)
        similarities = np.dot(all_product_embeddings, all_product_embeddings[prod_idx])
        all_similar = np.argsort(similarities)[::-1]
        
        for idx in all_similar:
            pid = all_product_ids[idx]
            if pid not in seen and len(final_recs) < top_k:
                final_recs.append(pid)
                seen.add(pid)
    
    # Enrich data
    detailed_recs = []
    for rec_id in final_recs:
        try:
            product_info = get_product_info(rec_id)
            detailed_recs.append(product_info)
        except:
            continue
    
    return {
        "product_id": product_id,
        "user_id": user_id,
        "mode": STRATEGY,
        "title": "Sản phẩm liên quan" if STRATEGY == "guest" else "Gợi ý cho bạn",
        "description": "Sản phẩm tương tự dựa trên đặc điểm" if STRATEGY == "guest" 
                      else "Kết hợp 'người dùng cũng thích' và đặc điểm tương tự",
        "strategy": {
            "mode": STRATEGY,
            "collaborative_weight": COLLAB_WEIGHT,
            "content_weight": CONTENT_WEIGHT,
            "collab_count": len(collab_recs),
            "content_count": len(content_recs),
            "method": f"Hybrid ({int(COLLAB_WEIGHT*100)}% collaborative + {int(CONTENT_WEIGHT*100)}% content)" 
                     if STRATEGY == "logged_in" else "Content-based only (100%)"
        },
        "total_recommendations": len(detailed_recs),
        "recommendations": detailed_recs
    }


@app.get("/product/{product_id}", tags=["Products"])
def get_product(product_id: str):
    """Get detailed product information"""
    return get_product_info(product_id)


@app.get("/stats", tags=["General"])
def get_statistics():
    """System statistics and model info"""
    stats = {
        "system": {
            "status": "operational" if (als_model and all_product_embeddings is not None) else "degraded",
            "api_version": "4.0-weighted",
            "features": ["weighted_interactions", "user_personalization", "item_similarity", "content_based"]
        },
        "collaborative_model": {
            "algorithm": "Weighted ALS (Matrix Factorization)",
            "status": "available" if als_model else "unavailable",
            "total_users": len(als_model.user_map) if als_model else 0,
            "total_items": len(als_model.item_map) if als_model else 0,
            "latent_factors": als_model.n_factors if als_model else 0,
            "alpha_confidence": als_model.alpha if als_model else None,
            "supports_user_personalization": als_model is not None and als_model.has_user_personalization(),
            "interaction_weights": {
                "click": 1,
                "add_to_cart": 2,
                "purchase": 3
            }
        },
        "content_model": {
            "algorithm": "PhoBERT + Cosine Similarity",
            "status": "available" if all_product_embeddings is not None else "unavailable",
            "total_products": len(all_product_ids) if all_product_ids else 0,
            "embedding_dimension": all_product_embeddings.shape[1] if all_product_embeddings is not None else 0
        },
        "product_catalog": {
            "total_products": len(products_df) if products_df is not None else 0
        }
    }
    
    return stats


@app.get("/health", tags=["General"])
def health_check():
    """Detailed health check for monitoring"""
    health = {
        "status": "healthy",
        "checks": {
            "als_model": als_model is not None,
            "user_factors_loaded": als_model is not None and als_model.has_user_personalization(),
            "phobert_embeddings": all_product_embeddings is not None,
            "product_data": products_df is not None
        }
    }
    
    if not all(health["checks"].values()):
        health["status"] = "unhealthy"
        health["message"] = "Some models are not loaded"
    
    return health


# === ERROR HANDLERS ===

@app.exception_handler(404)
def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": str(exc.detail) if hasattr(exc, 'detail') else "Resource not found",
        "status_code": 404
    }


@app.exception_handler(500)
def internal_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }


if __name__ == "__main__":
    print("  Homepage (personalized): http://localhost:8000/recommend/homepage?user_id=xxx")
    print("  Product detail (hybrid):  http://localhost:8000/recommend/product-detail/{product_id}?user_id=xxx")
    print("\nDocumentation: http://localhost:8000/docs")
    print("Health check:  http://localhost:8000/health")

    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )