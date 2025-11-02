"""
Recommend Service API - Production Implementation
Hybrid Recommendation System (ALS + PhoBERT)
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
from pydantic import BaseModel
import pickle
import numpy as np
import logging
from datetime import datetime
import asyncio
import uuid

from mf import AlternatingLeastSquares

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="Recommend Service",
    description="Hybrid Recommendation System API (ALS + PhoBERT)",
    version="1.0.0"
)

# ============================================
# GLOBAL STATE
# ============================================
class ModelState:
    """Global state for models"""
    def __init__(self):
        self.als_model: Optional[AlternatingLeastSquares] = None
        self.phobert_embeddings: Optional[np.ndarray] = None
        self.product_ids: Optional[List[str]] = None
        self.models_loaded: bool = False
        self.last_training_run: Dict = {}
        self.current_training_job: Optional[str] = None
        
model_state = ModelState()

# ============================================
# STARTUP - Load Models
# ============================================
@app.on_event("startup")
async def load_models():
    """Load models on service startup"""
    try:
        logger.info("Starting Recommend Service...")
        
        # Load ALS
        try:
            model_state.als_model = AlternatingLeastSquares.load("models/als_model.pkl")
            logger.info(f"ALS loaded: {len(model_state.als_model.user_map)} users, "
                       f"{len(model_state.als_model.item_map)} items")
        except FileNotFoundError:
            logger.warning("ALS model not found - CF disabled")
        
        # Load PhoBERT
        try:
            with open("data/phobert_embeddings.pkl", 'rb') as f:
                data = pickle.load(f)
            model_state.phobert_embeddings = data['embeddings']
            model_state.product_ids = data['product_ids']
            logger.info(f"PhoBERT loaded: {len(model_state.product_ids)} products")
        except FileNotFoundError:
            logger.warning("PhoBERT embeddings not found - Content-based disabled")
        
        if model_state.als_model or model_state.phobert_embeddings is not None:
            model_state.models_loaded = True
            logger.info("Service ready")
        else:
            logger.error("No models loaded")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_content_recommendations(product_id: str, top_k: int = 10) -> List[str]:
    """Get content-based recommendations (PhoBERT)"""
    if model_state.phobert_embeddings is None:
        raise HTTPException(status_code=503, detail="PhoBERT model unavailable")
    
    if product_id not in model_state.product_ids:
        raise HTTPException(status_code=404, 
                          detail=f"Product {product_id} not found in PhoBERT index")
    
    try:
        prod_idx = model_state.product_ids.index(product_id)
        target_vec = model_state.phobert_embeddings[prod_idx]
        similarities = np.dot(model_state.phobert_embeddings, target_vec)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        return [model_state.product_ids[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Content-based error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_collaborative_recommendations(user_id: str, top_k: int = 10) -> Optional[List[str]]:
    """Get personalized recommendations (ALS)"""
    if model_state.als_model is None:
        raise HTTPException(status_code=503, detail="ALS model unavailable")
    
    if not model_state.als_model.has_user_personalization():
        raise HTTPException(status_code=503, detail="User personalization unavailable")
    
    return model_state.als_model.predict_for_user(user_id, top_k=top_k)

def get_similar_items_collaborative(product_id: str, top_k: int = 10) -> Optional[List[str]]:
    """Get similar items using ALS (item-item similarity)"""
    if model_state.als_model is None:
        return None
    return model_state.als_model.get_similar_items(product_id, top_k=top_k)

def hybrid_merge(content_list: List[str], collab_list: Optional[List[str]], 
                 content_weight: float = 0.6, top_k: int = 10) -> List[str]:
    """Merge content-based and collaborative results"""
    if collab_list is None or len(collab_list) == 0:
        return content_list[:top_k]
    
    num_content = int(top_k * content_weight)
    num_collab = top_k - num_content
    
    result = []
    seen = set()
    
    # Add from content-based
    for pid in content_list[:num_content]:
        if pid not in seen:
            result.append(pid)
            seen.add(pid)
    
    # Add from collaborative
    for pid in collab_list[:num_collab]:
        if pid not in seen and len(result) < top_k:
            result.append(pid)
            seen.add(pid)
    
    # Fill remaining from content
    for pid in content_list[num_content:]:
        if pid not in seen and len(result) < top_k:
            result.append(pid)
            seen.add(pid)
    
    return result[:top_k]

def get_popular_fallback(top_k: int = 10) -> List[str]:
    """Fallback: return popular items"""
    if model_state.als_model is None:
        if model_state.product_ids:
            return model_state.product_ids[:top_k]
        return []
    
    # Estimate popularity from ALS item factors
    item_popularity = {}
    for item_id in model_state.als_model.item_map.keys():
        item_idx = model_state.als_model.item_map[item_id]
        popularity = np.linalg.norm(model_state.als_model.item_factors[item_idx])
        item_popularity[item_id] = popularity
    
    sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in sorted_items[:top_k]]

# ============================================
# API ENDPOINTS - Recommendations
# ============================================

@app.get("/recommend/homepage")
async def recommend_homepage(
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations")
) -> List[str]:
    """
    API 3.2: Homepage recommendations
    - With user_id: Personalized (ALS)
    - Without user_id: Popular items (fallback)
    """
    if not model_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Case 1: Logged-in user -> Collaborative Filtering
    if user_id:
        try:
            recs = get_collaborative_recommendations(user_id, top_k)
            if recs:
                return recs
        except Exception as e:
            logger.warning(f"Collaborative failed for user {user_id}: {e}")
    
    # Case 2: Guest or Cold Start -> Popular fallback
    return get_popular_fallback(top_k)


@app.get("/recommend/product-detail/{product_id}")
async def recommend_product_detail(
    product_id: str,
    user_id: Optional[str] = Query(None, description="User ID for hybrid strategy"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations")
) -> List[str]:
    """
    API 3.3: Product detail page recommendations
    - Without user_id: 100% Content-based (PhoBERT)
    - With user_id: Hybrid (60% Content + 40% Collaborative)
    """
    if not model_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Content-based (required)
    try:
        content_recs = get_content_recommendations(product_id, top_k=top_k)
    except HTTPException as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, 
                              detail=f"Product {product_id} not found in models")
        raise
    
    # Case 1: Guest -> 100% Content-based
    if not user_id:
        return content_recs
    
    # Case 2: Logged-in -> Hybrid Strategy
    try:
        collab_recs = get_similar_items_collaborative(product_id, top_k=top_k)
        
        if collab_recs is None:
            return content_recs
        
        # Merge with 60% content, 40% collaborative
        return hybrid_merge(content_recs, collab_recs, content_weight=0.6, top_k=top_k)
        
    except Exception as e:
        logger.warning(f"Collaborative failed, fallback to content: {e}")
        return content_recs

# ============================================
# API ENDPOINTS - Training & Admin
# ============================================

class TrainingRequest(BaseModel):
    force_retrain_all: bool = False
    model_version_tag: Optional[str] = None

@app.post("/train", status_code=202)
async def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    API 5.2.1: Trigger training pipeline
    Returns 202 Accepted and runs training in background
    """
    if model_state.current_training_job:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "A training process is already running",
                "running_job_id": model_state.current_training_job
            }
        )
    
    job_id = str(uuid.uuid4())
    model_state.current_training_job = job_id
    
    model_state.last_training_run = {
        "job_id": job_id,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "status": "RUNNING",
        "triggered_by": "API",
        "force_retrain_all": request.force_retrain_all
    }
    
    logger.info(f"Training job {job_id} started")
    background_tasks.add_task(run_training_pipeline, job_id, request)
    
    return {
        "status": "Training process started",
        "job_id": job_id
    }

async def run_training_pipeline(job_id: str, request: TrainingRequest):
    """Background task for training pipeline"""
    try:
        logger.info(f"Job {job_id}: Starting training pipeline")
        
        # TODO: Implement full training pipeline
        # 1. Fetch data from Product Service & Order Service
        # 2. Train ALS model
        # 3. Generate PhoBERT embeddings
        # 4. Hot-swap models
        
        await asyncio.sleep(5)  # Simulate training
        
        model_state.last_training_run.update({
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "status": "COMPLETED_SUCCESS",
            "duration_seconds": 5
        })
        model_state.current_training_job = None
        
        logger.info(f"Job {job_id}: Training completed")
        
    except Exception as e:
        logger.error(f"Job {job_id}: Training failed - {e}")
        model_state.last_training_run.update({
            "status": "FAILED",
            "error": str(e)
        })
        model_state.current_training_job = None


@app.get("/admin/metrics")
async def get_metrics():
    """
    API 6.2.1: Get metrics and model status
    """
    return {
        "service_status": "ONLINE" if model_state.models_loaded else "DEGRADED",
        "last_training_run": model_state.last_training_run or {
            "status": "NO_TRAINING_YET"
        },
        "active_models": {
            "als_model": {
                "status": "loaded" if model_state.als_model else "unavailable",
                "users": len(model_state.als_model.user_map) if model_state.als_model else 0,
                "items": len(model_state.als_model.item_map) if model_state.als_model else 0,
                "latent_factors": model_state.als_model.n_factors if model_state.als_model else 0
            },
            "phobert_model": {
                "status": "loaded" if model_state.phobert_embeddings is not None else "unavailable",
                "total_items": len(model_state.product_ids) if model_state.product_ids else 0,
                "embedding_dim": model_state.phobert_embeddings.shape[1] if model_state.phobert_embeddings is not None else 0
            }
        }
    }


@app.get("/health")
async def health_check():
    """
    API 6.3.1: Health check for Load Balancer
    """
    if not model_state.models_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "UNHEALTHY",
                "error": "Models not loaded or initializing"
            }
        )
    
    models_loaded = []
    if model_state.als_model:
        models_loaded.append("als")
    if model_state.phobert_embeddings is not None:
        models_loaded.append("phobert")
    
    return {
        "status": "HEALTHY",
        "models_loaded": models_loaded
    }

# ROOT ENDPOINT

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Recommend Service",
        "version": "1.0.0",
        "status": "online" if model_state.models_loaded else "initializing",
        "endpoints": {
            "homepage": "/recommend/homepage",
            "product_detail": "/recommend/product-detail/{product_id}",
            "training": "/train",
            "metrics": "/admin/metrics",
            "health": "/health"
        },
        "documentation": "/docs"
    }

# RUN SERVER

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )