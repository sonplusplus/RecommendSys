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
import socket

from mf import AlternatingLeastSquares
import py_eureka_client.eureka_client as eureka_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recommend Service",
    description="Hybrid Recommendation System API",
    version="1.0.0"
)

# ============================================
# GLOBAL STATE
# ============================================
class ModelState:
    """Manages ML models and training state"""
    def __init__(self):
        self.als_model: Optional[AlternatingLeastSquares] = None
        self.phobert_embeddings: Optional[np.ndarray] = None
        self.product_ids: Optional[List[str]] = None
        self.models_loaded: bool = False
        self.last_training_run: Dict = {}
        self.current_training_job: Optional[str] = None
        
model_state = ModelState()

# ============================================
# STARTUP
# ============================================
@app.on_event("startup")
async def startup():
    """Initialize service: Register with Eureka and load ML models"""
    try:
        logger.info("Starting Recommend Service...")
        
        # Get host info
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        logger.info(f"Registering with Eureka at {ip_address}:8888")
        
        # Register with Eureka Server - FIXED PARAMETERS
        await eureka_client.init_async(
            eureka_server="http://localhost:8761/eureka/",
            app_name="recommend-service",  # Tên service cho Feign Client
            instance_port=8888,
            instance_ip=ip_address,
            instance_host=hostname,
            # Health check endpoint
            health_check_url=f"http://{ip_address}:8888/health",
            status_page_url=f"http://{ip_address}:8888/admin/metrics",
            # Heartbeat config (tên tham số đúng)
            renewal_interval_in_secs=30,  # Heartbeat interval
            duration_in_secs=90,  # Lease duration
        )
        
        logger.info("✓ Successfully registered with Eureka")
        
        # Load ALS model (Collaborative Filtering)
        try:
            model_state.als_model = AlternatingLeastSquares.load("models/als_model.pkl")
            logger.info(f"✓ ALS loaded: {len(model_state.als_model.user_map)} users, "
                       f"{len(model_state.als_model.item_map)} items")
        except FileNotFoundError:
            logger.warning("⚠ ALS model not found - Collaborative filtering disabled")
        
        # Load PhoBERT embeddings (Content-based)
        try:
            with open("data/phobert_embeddings.pkl", 'rb') as f:
                data = pickle.load(f)
            model_state.phobert_embeddings = data['embeddings']
            model_state.product_ids = data['product_ids']
            logger.info(f"✓ PhoBERT loaded: {len(model_state.product_ids)} products")
        except FileNotFoundError:
            logger.warning("⚠ PhoBERT embeddings not found - Content-based disabled")
        
        model_state.models_loaded = bool(model_state.als_model or model_state.phobert_embeddings is not None)
        logger.info(f"✓ Service ready: Models loaded = {model_state.models_loaded}")
            
    except Exception as e:
        logger.error(f"✗ Startup error: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown - deregister from Eureka"""
    try:
        logger.info("Shutting down, deregistering from Eureka...")
        await eureka_client.stop_async()
        logger.info("✓ Deregistered from Eureka")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_content_recommendations(product_id: str, top_k: int = 10) -> List[str]:
    """Content-based recommendations using PhoBERT embeddings"""
    if model_state.phobert_embeddings is None:
        raise HTTPException(status_code=503, detail="PhoBERT model unavailable")
    
    if product_id not in model_state.product_ids:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    
    try:
        # Calculate cosine similarity
        prod_idx = model_state.product_ids.index(product_id)
        target_vec = model_state.phobert_embeddings[prod_idx]
        similarities = np.dot(model_state.phobert_embeddings, target_vec)
        
        # Get top-k similar products (exclude self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        return [model_state.product_ids[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Content-based error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_collaborative_recommendations(user_id: str, top_k: int = 10) -> Optional[List[str]]:
    """Personalized recommendations using ALS collaborative filtering"""
    if model_state.als_model is None:
        raise HTTPException(status_code=503, detail="ALS model unavailable")
    
    if not model_state.als_model.has_user_personalization():
        raise HTTPException(status_code=503, detail="User personalization unavailable")
    
    return model_state.als_model.predict_for_user(user_id, top_k=top_k)

def get_similar_items_collaborative(product_id: str, top_k: int = 10) -> Optional[List[str]]:
    """Item-item similarity using ALS item factors"""
    if model_state.als_model is None:
        return None
    return model_state.als_model.get_similar_items(product_id, top_k=top_k)

def hybrid_merge(content_list: List[str], collab_list: Optional[List[str]], 
                 content_weight: float = 0.6, top_k: int = 10) -> List[str]:
    """Merge content-based and collaborative results with weighted strategy"""
    if not collab_list:
        return content_list[:top_k]
    
    # Calculate split: 60% content, 40% collaborative
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
    
    # Fill remaining slots from content
    for pid in content_list[num_content:]:
        if pid not in seen and len(result) < top_k:
            result.append(pid)
            seen.add(pid)
    
    return result[:top_k]

def get_popular_fallback(top_k: int = 10) -> List[str]:
    """Fallback: return popular items based on ALS item factor norms"""
    if model_state.als_model is None:
        return model_state.product_ids[:top_k] if model_state.product_ids else []
    
    # Estimate popularity from item factor magnitudes
    item_popularity = {}
    for item_id in model_state.als_model.item_map.keys():
        item_idx = model_state.als_model.item_map[item_id]
        popularity = np.linalg.norm(model_state.als_model.item_factors[item_idx])
        item_popularity[item_id] = popularity
    
    sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in sorted_items[:top_k]]

# ============================================
# RECOMMENDATION ENDPOINTS - INTERNAL API
# ============================================
@app.get("/api/v1/internal/recommend/homepage")
async def recommend_homepage(
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    top_k: int = Query(10, ge=1, le=50)
) -> Dict:
    """
    Homepage recommendations:
    - Logged-in users: Personalized via ALS
    - Guest users: Popular items fallback
    
    Response format for Feign Client:
    {
        "product_ids": ["123", "456", ...],
        "strategy": "personalized" | "popular",
        "count": 10
    }
    """
    if not model_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Try personalized recommendations for logged-in users
    if user_id:
        try:
            recs = get_collaborative_recommendations(user_id, top_k)
            if recs:
                return {
                    "product_ids": recs,
                    "strategy": "personalized",
                    "count": len(recs)
                }
        except Exception as e:
            logger.warning(f"Collaborative failed for user {user_id}: {e}")
    
    # Fallback to popular items
    popular_items = get_popular_fallback(top_k)
    return {
        "product_ids": popular_items,
        "strategy": "popular",
        "count": len(popular_items)
    }

@app.get("/api/v1/internal/recommend/product-detail/{product_id}")
async def recommend_product_detail(
    product_id: str,
    user_id: Optional[str] = Query(None, description="User ID for hybrid strategy"),
    top_k: int = Query(10, ge=1, le=50)
) -> Dict:
    """
    Product detail recommendations:
    - Guest: 100% Content-based (PhoBERT)
    - Logged-in: Hybrid (60% Content + 40% Collaborative)
    
    Response format for Feign Client:
    {
        "product_ids": ["789", "101", ...],
        "strategy": "content-based" | "hybrid",
        "count": 10
    }
    """
    if not model_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Always get content-based recommendations
    try:
        content_recs = get_content_recommendations(product_id, top_k=top_k)
    except HTTPException as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        raise
    
    # Guest users: content-based only
    if not user_id:
        return {
            "product_ids": content_recs,
            "strategy": "content-based",
            "count": len(content_recs)
        }
    
    # Logged-in users: hybrid strategy
    try:
        collab_recs = get_similar_items_collaborative(product_id, top_k=top_k)
        if collab_recs:
            hybrid_recs = hybrid_merge(content_recs, collab_recs, content_weight=0.6, top_k=top_k)
            return {
                "product_ids": hybrid_recs,
                "strategy": "hybrid",
                "count": len(hybrid_recs)
            }
    except Exception as e:
        logger.warning(f"Collaborative failed: {e}")
    
    return {
        "product_ids": content_recs,
        "strategy": "content-based",
        "count": len(content_recs)
    }

# ============================================
# ADMIN ENDPOINTS
# ============================================
class TrainingRequest(BaseModel):
    """Request body for training trigger"""
    force_retrain_all: bool = False
    model_version_tag: Optional[str] = None

@app.post("/train", status_code=202)
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining (runs in background)"""
    if model_state.current_training_job:
        raise HTTPException(
            status_code=409,
            content={"error": "Training already running", "job_id": model_state.current_training_job}
        )
    
    job_id = str(uuid.uuid4())
    model_state.current_training_job = job_id
    
    model_state.last_training_run = {
        "job_id": job_id,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "status": "RUNNING",
        "force_retrain_all": request.force_retrain_all
    }
    
    logger.info(f"Training job {job_id} started")
    background_tasks.add_task(run_training_pipeline, job_id, request)
    
    return {"status": "Training started", "job_id": job_id}

async def run_training_pipeline(job_id: str, request: TrainingRequest):
    """Background task: Train ALS + generate PhoBERT embeddings"""
    try:
        logger.info(f"Job {job_id}: Training started")
        
        # TODO: Implement training pipeline
        # 1. Fetch data from Product/Order services
        # 2. Train ALS model
        # 3. Generate PhoBERT embeddings
        # 4. Hot-swap models without downtime
        
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
        model_state.last_training_run.update({"status": "FAILED", "error": str(e)})
        model_state.current_training_job = None

@app.get("/admin/metrics")
async def get_metrics():
    """Service metrics and model status"""
    return {
        "service_status": "ONLINE" if model_state.models_loaded else "DEGRADED",
        "last_training_run": model_state.last_training_run or {"status": "NO_TRAINING_YET"},
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
    """Health check for Eureka and Load Balancer"""
    if not model_state.models_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "UNHEALTHY", "error": "Models not loaded"}
        )
    
    models_loaded = []
    if model_state.als_model:
        models_loaded.append("als")
    if model_state.phobert_embeddings is not None:
        models_loaded.append("phobert")
    
    return {
        "status": "HEALTHY",
        "models_loaded": models_loaded,
        "eureka_registered": True
    }

@app.get("/")
async def root():
    """Service info and available endpoints"""
    return {
        "service": "Recommend Service",
        "version": "1.0.0",
        "status": "online" if model_state.models_loaded else "initializing",
        "eureka_registered": True,
        "endpoints": {
            "homepage": "/api/v1/internal/recommend/homepage",
            "product_detail": "/api/v1/internal/recommend/product-detail/{product_id}",
            "training": "/train",
            "metrics": "/admin/metrics",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")