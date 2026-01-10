# DOCKER VOLUME DATA PERSISTENCE
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import pickle
import numpy as np
import logging
from datetime import datetime
import asyncio
import uuid
import socket
import aiohttp
from pathlib import Path
import json
import pandas as pd
from mf import AlternatingLeastSquares
import py_eureka_client.eureka_client as eureka_client

# LOGGING CONFIGURATION

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# CONFIGURATION
import os

class ServiceConfig:
    """Configuration for service dependencies and model paths"""
    EUREKA_SERVER = os.getenv("EUREKA_SERVER", "http://eureka-server:8761/eureka")
    SERVICE_NAME = "recommend-service"
    SERVICE_PORT = 8086
    PRODUCT_SERVICE_URL = os.getenv("PRODUCT_SERVICE_URL", "http://api-gateway:8080")
    ADMIN_SERVICE_URL = os.getenv("ADMIN_SERVICE_URL", "http://api-gateway:8080")
    PRODUCT_EXPORT_ENDPOINT = "/api/v1/internal/admin/export/products"
    INTERACTION_EXPORT_ENDPOINT = "/api/v1/internal/admin/export/interactions"
    INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "a-very-secret-key-for-internal-communication")
    API_KEY_HEADER = "X-API-KEY"
    MODEL_DIR = Path("/app/models")
    DATA_DIR = Path("/app/data")
    ALS_MODEL_PATH = MODEL_DIR / "als_model.pkl"
    PHOBERT_EMBEDDINGS_PATH = MODEL_DIR / "phobert_embeddings.pkl"
    BASE_INTERACTIONS_CSV = DATA_DIR / "interactions.csv"
    BASE_PRODUCTS_CSV = DATA_DIR / "products.csv"
    ACCUMULATED_INTERACTIONS_CSV = DATA_DIR / "accumulated_interactions.csv"
    ACCUMULATED_PRODUCTS_CSV = DATA_DIR / "accumulated_products.csv"
    ALS_N_FACTORS = int(os.getenv("ALS_N_FACTORS", "50"))
    ALS_REGULARIZATION = float(os.getenv("ALS_REGULARIZATION", "0.01"))
    ALS_N_ITERATIONS = int(os.getenv("ALS_N_ITERATIONS", "15"))
    ALS_ALPHA = float(os.getenv("ALS_ALPHA", "40"))
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    REQUEST_TIMEOUT = 60
    INCREMENTAL_THRESHOLD = 1_000_000      # 1M interactions
    FULL_RETRAIN_THRESHOLD = 2_000_000     # 2M interactions
    WARMSTART_WEIGHT_OLD = 0.7             # 70% old model
    WARMSTART_WEIGHT_NEW = 0.3            # 30% new model

config = ServiceConfig()

# ============================================
# PYDANTIC MODELS
# ============================================
class ProductData(BaseModel):
    product_id: str
    name: str
    description: str
    brand: str
    category: str

class InteractionData(BaseModel):
    user_id: str
    product_id: str
    weight: float
    type: str

class TrainingRequest(BaseModel):
    force_retrain_all: bool = Field(default=False)
    model_version_tag: Optional[str] = Field(default=None)

class RecommendationResponse(BaseModel):
    product_ids: List[str]
    strategy: str
    count: int

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
        self.last_training_run: Dict[str, Any] = {}
        self.current_training_job: Optional[str] = None
        self._lock = asyncio.Lock()

    async def update_models(self, new_als_model: Optional[AlternatingLeastSquares],
                           new_phobert_embeddings: Optional[np.ndarray],
                           new_product_ids: Optional[List[str]]):
        """Thread-safe hot-swap of models"""
        async with self._lock:
            if new_als_model:
                self.als_model = new_als_model
                logger.info("ALS model hot-swapped")

            if new_phobert_embeddings is not None and new_product_ids:
                self.phobert_embeddings = new_phobert_embeddings
                self.product_ids = new_product_ids
                logger.info("PhoBERT embeddings hot-swapped")

            self.models_loaded = bool(self.als_model or self.phobert_embeddings is not None)

model_state = ModelState()

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================
app = FastAPI(
    title="Recommend Service",
    description="Hybrid Recommendation System API (ALS + PhoBERT)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# HTTP CLIENT WITH RETRY LOGIC
# ============================================
class HttpClient:
    """Async HTTP client with retry logic"""

    @staticmethod
    async def fetch_with_retry(url: str, max_retries: int = config.MAX_RETRIES,
                               timeout: int = config.REQUEST_TIMEOUT) -> Dict:
        last_error = None

        for attempt in range(max_retries):
            try:
                timeout_obj = aiohttp.ClientTimeout(total=timeout)
                headers = {config.API_KEY_HEADER: config.INTERNAL_API_KEY}

                async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                    logger.info(f"HTTP GET {url} (attempt {attempt + 1}/{max_retries})")

                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Successfully fetched from {url}")
                            return data
                        else:
                            error_text = await response.text()
                            logger.warning(f"HTTP {response.status} from {url}: {error_text}")
                            last_error = f"HTTP {response.status}: {error_text}"

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")

            except aiohttp.ClientError as e:
                last_error = str(e)
                logger.warning(f"Client error fetching {url}: {e}")

            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error fetching {url}: {e}")

            if attempt < max_retries - 1:
                wait_time = config.RETRY_DELAY * (2 ** attempt)
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        error_msg = f"Failed to fetch {url} after {max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)

http_client = HttpClient()

# ============================================
# DATA LOADING & MERGING STRATEGY
# ============================================

async def fetch_and_merge_all_data() -> tuple[List[ProductData], List[InteractionData]]:
    """
    STRATEGY: LUÔN KẾT HỢP 3 NGUỒN DATA
    
    1. BASE CSV (data giả lập - luôn load)
    2. ACCUMULATED CSV (data từ lần train trước)
    3. API DATA (data mới từ Postman test)
    
    Returns:
        Tuple of (merged_products, merged_interactions)
    """
    logger.info("=" * 70)
    logger.info("📊 LOADING DATA FROM ALL SOURCES")
    logger.info("=" * 70)

    all_interactions = []
    all_products = []

    # ============================================
    # SOURCE 1: BASE CSV (Data giả lập - LUÔN LOAD)
    # ============================================
    base_interactions_count = 0
    base_products_count = 0
    
    if config.BASE_INTERACTIONS_CSV.exists():
        try:
            logger.info(f"📂 Loading BASE interactions from: {config.BASE_INTERACTIONS_CSV}")
            base_df = pd.read_csv(config.BASE_INTERACTIONS_CSV)
            
            for _, row in base_df.iterrows():
                all_interactions.append(InteractionData(
                    user_id=str(row['user_id']),
                    product_id=str(row['product_id']),
                    weight=float(row.get('weight', 1.0)),
                    type=str(row.get('type', 'BASE'))
                ))
            
            base_interactions_count = len(all_interactions)
            logger.info(f"✅ Loaded {base_interactions_count} BASE interactions (giả lập)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load BASE interactions: {e}")
            logger.warning("⚠️  Training will continue with API data only")
    else:
        logger.warning(f"⚠️  BASE interactions not found at {config.BASE_INTERACTIONS_CSV}")
        logger.warning("⚠️  Make sure to mount CSV files to /app/data volume")

    if config.BASE_PRODUCTS_CSV.exists():
        try:
            logger.info(f"📂 Loading BASE products from: {config.BASE_PRODUCTS_CSV}")
            base_prod_df = pd.read_csv(config.BASE_PRODUCTS_CSV)
            
            for _, row in base_prod_df.iterrows():
                all_products.append(ProductData(
                    product_id=str(row['product_id']),
                    name=str(row.get('name', row.get('product_name', ''))),
                    description=str(row.get('description', '')),
                    brand=str(row.get('brand', '')),
                    category=str(row.get('category', ''))
                ))
            
            base_products_count = len(all_products)
            logger.info(f"✅ Loaded {base_products_count} BASE products (giả lập)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load BASE products: {e}")
    else:
        logger.warning(f"⚠️  BASE products not found at {config.BASE_PRODUCTS_CSV}")

    # ============================================
    # SOURCE 2: ACCUMULATED DATA (từ lần train trước)
    # ============================================
    accumulated_interactions_count = 0
    accumulated_products_count = 0
    
    if config.ACCUMULATED_INTERACTIONS_CSV.exists():
        try:
            logger.info(f"📂 Loading ACCUMULATED interactions from previous runs")
            acc_df = pd.read_csv(config.ACCUMULATED_INTERACTIONS_CSV)
            
            for _, row in acc_df.iterrows():
                all_interactions.append(InteractionData(
                    user_id=str(row['user_id']),
                    product_id=str(row['product_id']),
                    weight=float(row['weight']),
                    type=str(row.get('type', 'ACCUMULATED'))
                ))
            
            accumulated_interactions_count = len(all_interactions) - base_interactions_count
            logger.info(f"✅ Loaded {accumulated_interactions_count} ACCUMULATED interactions")
            
        except Exception as e:
            logger.warning(f"Could not load accumulated interactions: {e}")

    if config.ACCUMULATED_PRODUCTS_CSV.exists():
        try:
            logger.info(f"📂 Loading ACCUMULATED products from previous runs")
            acc_prod_df = pd.read_csv(config.ACCUMULATED_PRODUCTS_CSV)
            
            for _, row in acc_prod_df.iterrows():
                all_products.append(ProductData(
                    product_id=str(row['product_id']),
                    name=str(row['name']),
                    description=str(row.get('description', '')),
                    brand=str(row.get('brand', '')),
                    category=str(row.get('category', ''))
                ))
            
            accumulated_products_count = len(all_products) - base_products_count
            logger.info(f"✅ Loaded {accumulated_products_count} ACCUMULATED products")
            
        except Exception as e:
            logger.warning(f"Could not load accumulated products: {e}")

    # ============================================
    # SOURCE 3: API DATA (data mới từ Postman)
    # ============================================
    api_interactions_count = 0
    api_products_count = 0
    
    try:
        # Fetch products from Product Service
        product_url = f"{config.PRODUCT_SERVICE_URL}{config.PRODUCT_EXPORT_ENDPOINT}"
        logger.info(f"🌐 Fetching NEW products from API: {product_url}")
        products_raw = await http_client.fetch_with_retry(product_url)
        
        api_products = [ProductData(**p) for p in products_raw]
        all_products.extend(api_products)
        api_products_count = len(api_products)
        logger.info(f"✅ Fetched {api_products_count} products from API")

        # Fetch interactions from Admin Service
        interaction_url = f"{config.ADMIN_SERVICE_URL}{config.INTERACTION_EXPORT_ENDPOINT}"
        logger.info(f"🌐 Fetching NEW interactions from API: {interaction_url}")
        interactions_raw = await http_client.fetch_with_retry(interaction_url)
        
        api_interactions = [InteractionData(**i) for i in interactions_raw]
        all_interactions.extend(api_interactions)
        api_interactions_count = len(api_interactions)
        logger.info(f"✅ Fetched {api_interactions_count} interactions from API")

    except Exception as e:
        logger.error(f"❌ Failed to fetch from API: {e}")
        logger.warning("⚠️  Training will continue with CSV data only")

    # ============================================
    # MERGE & DEDUPLICATE
    # ============================================
    logger.info("=" * 70)
    logger.info("🔄 MERGING & DEDUPLICATING DATA")
    logger.info("=" * 70)

    # Deduplicate products (keep latest by product_id)
    products_dict = {}
    for product in all_products:
        products_dict[product.product_id] = product
    merged_products = list(products_dict.values())

    # Deduplicate interactions (sum weights for same user-product)
    interactions_dict = {}
    for interaction in all_interactions:
        key = (interaction.user_id, interaction.product_id)
        if key in interactions_dict:
            interactions_dict[key].weight += interaction.weight
        else:
            interactions_dict[key] = interaction
    
    merged_interactions = list(interactions_dict.values())

    # ============================================
    # SUMMARY
    # ============================================
    logger.info("=" * 70)
    logger.info("📊 DATA MERGE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"SOURCE 1 - BASE CSV (giả lập):")
    logger.info(f"  └─ Interactions: {base_interactions_count:,}")
    logger.info(f"  └─ Products: {base_products_count:,}")
    logger.info(f"")
    logger.info(f"SOURCE 2 - ACCUMULATED (từ lần trước):")
    logger.info(f"  └─ Interactions: {accumulated_interactions_count:,}")
    logger.info(f"  └─ Products: {accumulated_products_count:,}")
    logger.info(f"")
    logger.info(f"SOURCE 3 - API (mới từ Postman):")
    logger.info(f"  └─ Interactions: {api_interactions_count:,}")
    logger.info(f"  └─ Products: {api_products_count:,}")
    logger.info(f"")
    logger.info(f"📊 FINAL MERGED DATA (after deduplication):")
    logger.info(f"  └─ Total Interactions: {len(merged_interactions):,}")
    logger.info(f"  └─ Total Products: {len(merged_products):,}")
    logger.info(f"  └─ Unique Users: {len(set(i.user_id for i in merged_interactions)):,}")
    logger.info("=" * 70)

    # ✅ SAVE ACCUMULATED DATA (CHỈ LƯU API DATA MỚI - KHÔNG LƯU BASE)
    # Để lần sau không bị duplicate BASE data
    save_accumulated_data(api_interactions, api_products)

    return merged_products, merged_interactions


def save_accumulated_data(api_interactions: List[InteractionData], api_products: List[ProductData]):
    """
    Lưu CHỈ API data vào accumulated files
    
    Strategy: Append API data vào accumulated, BASE CSV giữ nguyên
    """
    try:
        # Load existing accumulated data
        existing_interactions = []
        if config.ACCUMULATED_INTERACTIONS_CSV.exists():
            acc_df = pd.read_csv(config.ACCUMULATED_INTERACTIONS_CSV)
            existing_interactions = [
                InteractionData(
                    user_id=str(row['user_id']),
                    product_id=str(row['product_id']),
                    weight=float(row['weight']),
                    type=str(row.get('type', 'API'))
                )
                for _, row in acc_df.iterrows()
            ]
        
        # Merge with new API data
        all_accumulated = existing_interactions + api_interactions
        
        # Deduplicate
        interactions_dict = {}
        for interaction in all_accumulated:
            key = (interaction.user_id, interaction.product_id)
            if key in interactions_dict:
                interactions_dict[key].weight += interaction.weight
            else:
                interactions_dict[key] = interaction
        
        # Save
        interactions_df = pd.DataFrame([
            {
                'user_id': i.user_id,
                'product_id': i.product_id,
                'weight': i.weight,
                'type': i.type
            }
            for i in interactions_dict.values()
        ])
        interactions_df.to_csv(config.ACCUMULATED_INTERACTIONS_CSV, index=False)
        logger.info(f"✅ Saved {len(interactions_df)} accumulated interactions to {config.ACCUMULATED_INTERACTIONS_CSV}")
        
        # Similar for products
        existing_products = []
        if config.ACCUMULATED_PRODUCTS_CSV.exists():
            prod_df = pd.read_csv(config.ACCUMULATED_PRODUCTS_CSV)
            existing_products = [
                ProductData(
                    product_id=str(row['product_id']),
                    name=str(row['name']),
                    description=str(row.get('description', '')),
                    brand=str(row.get('brand', '')),
                    category=str(row.get('category', ''))
                )
                for _, row in prod_df.iterrows()
            ]
        
        all_products = existing_products + api_products
        
        # Deduplicate products
        products_dict = {p.product_id: p for p in all_products}
        
        products_df = pd.DataFrame([
            {
                'product_id': p.product_id,
                'name': p.name,
                'description': p.description,
                'brand': p.brand,
                'category': p.category
            }
            for p in products_dict.values()
        ])
        products_df.to_csv(config.ACCUMULATED_PRODUCTS_CSV, index=False)
        logger.info(f"✅ Saved {len(products_df)} accumulated products to {config.ACCUMULATED_PRODUCTS_CSV}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save accumulated data: {e}")



# SIMPLIFIED TRAINING PIPELINE




async def run_training_pipeline(job_id: str, request: TrainingRequest):

    start_time = datetime.utcnow()
    logger.info("=" * 70)
    logger.info(f"🚀 TRAINING JOB {job_id} STARTED")
    logger.info("=" * 70)

    try:
        # ✅ LOAD ALL DATA
        products, interactions = await fetch_and_merge_all_data()

        if len(products) == 0:
            raise ValueError("❌ No products available")
        if len(interactions) == 0:
            raise ValueError("❌ No interactions available")

        # Tách lại 3 nguồn để analyze
        base_interactions = [i for i in interactions if i.type == 'BASE']
        accumulated_interactions = [i for i in interactions if i.type == 'ACCUMULATED']
        api_interactions = [i for i in interactions if i.type == 'API']
        
        analysis = analyze_dataset_size(
            base_interactions,
            accumulated_interactions,
            api_interactions
        )
        
        logger.info("=" * 70)
        logger.info("📊 DATASET SIZE ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"BASE data:        {analysis['base_size']:,} interactions")
        logger.info(f"ACCUMULATED data: {analysis['accumulated_size']:,} interactions")
        logger.info(f"API data (new):   {analysis['api_size']:,} interactions")
        logger.info(f"TOTAL:            {analysis['total_size']:,} interactions")
        logger.info("")
        logger.info(f"🎯 RECOMMENDED STRATEGY: {analysis['recommended_strategy'].upper()}")
        logger.info(f"📝 Reason: {analysis['reason']}")
        logger.info("=" * 70)


        strategy = analysis['recommended_strategy']
        
        if request.force_retrain_all:
            strategy = 'full_retrain'
            logger.info("⚠️  User forced FULL RETRAIN (override auto-decision)")

  
        logger.info("=" * 70)
        logger.info(f"🤖 TRAINING ALS MODEL - Strategy: {strategy.upper()}")
        logger.info("=" * 70)
        
        loop = asyncio.get_event_loop()
        old_model = None
        
        if strategy == 'skip':
            # Giữ nguyên model hiện tại
            logger.info("⏭️  Skipping training - API data too small")
            new_als_model = model_state.als_model
            training_skipped = True
        
        elif strategy == 'full_retrain':
            # Full retrain từ đầu
            new_als_model = await loop.run_in_executor(
                None, 
                train_als_model, 
                interactions
            )
            training_skipped = False
        
        elif strategy == 'warmstart':
            # Warmstart từ model cũ
            if config.ALS_MODEL_PATH.exists():
                try:
                    old_model = AlternatingLeastSquares.load(str(config.ALS_MODEL_PATH))
                    logger.info(f"✅ Loaded old model for warmstart")
                except Exception as e:
                    logger.warning(f"Could not load old model: {e}. Fallback to full retrain.")
                    old_model = None
            
            if old_model:
                new_als_model = await loop.run_in_executor(
                    None, 
                    train_als_model_with_warmstart, 
                    interactions,
                    old_model
                )
            else:
                new_als_model = await loop.run_in_executor(
                    None, 
                    train_als_model, 
                    interactions
                )
            training_skipped = False
        
        elif strategy == 'incremental':
            # Incremental learning (future implementation)
            logger.warning("⚠️  Incremental learning chưa implement. Fallback to warmstart.")
            
            if config.ALS_MODEL_PATH.exists():
                old_model = AlternatingLeastSquares.load(str(config.ALS_MODEL_PATH))
            
            new_als_model = await loop.run_in_executor(
                None, 
                train_als_model_with_warmstart if old_model else train_als_model, 
                interactions,
                old_model
            )
            training_skipped = False
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        should_retrain_phobert = request.force_retrain_all or not config.PHOBERT_EMBEDDINGS_PATH.exists()
        
        if should_retrain_phobert:
            logger.info("🤖 GENERATING PHOBERT EMBEDDINGS")
            new_embeddings, new_product_ids = await loop.run_in_executor(
                None, 
                generate_phobert_embeddings, 
                products
            )
        else:
            logger.info("📂 Reusing existing PhoBERT embeddings")
            with open(config.PHOBERT_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
            new_embeddings = data['embeddings']
            new_product_ids = data['product_ids']

        if not training_skipped:
            logger.info("💾 SAVING MODELS")
            new_als_model.save(str(config.ALS_MODEL_PATH))
            logger.info(f"✅ ALS model saved")
        
        with open(config.PHOBERT_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump({
                'embeddings': new_embeddings,
                'product_ids': new_product_ids,
                'model_name': 'vinai/phobert-base',
                'normalized': True,
                'embedding_dim': new_embeddings.shape[1],
                'created_at': datetime.utcnow().isoformat() + "Z"
            }, f)

        # ✅ HOT-SWAP
        await model_state.update_models(
            new_als_model=new_als_model,
            new_phobert_embeddings=new_embeddings,
            new_product_ids=new_product_ids
        )

        # Update status
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        model_state.last_training_run.update({
            "completed_at": end_time.isoformat() + "Z",
            "status": "COMPLETED_SUCCESS",
            "duration_seconds": duration,
            "strategy": strategy,  # 🆕
            "dataset_analysis": analysis,  # 🆕
            "metrics": {
                "total_products": len(products),
                "total_interactions": len(interactions),
                "base_interactions": analysis['base_size'],
                "accumulated_interactions": analysis['accumulated_size'],
                "api_interactions": analysis['api_size'],
                "unique_users": len(set(i.user_id for i in interactions)),
                "als_users": len(new_als_model.user_map),
                "als_items": len(new_als_model.item_map),
                "phobert_retrained": should_retrain_phobert,
                "training_skipped": training_skipped  # 🆕
            }
        })
        model_state.current_training_job = None

        logger.info("=" * 70)
        logger.info(f"✅ TRAINING JOB {job_id} COMPLETED in {duration:.2f}s")
        logger.info(f"📊 Strategy used: {strategy.upper()}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ TRAINING FAILED: {e}", exc_info=True)
        model_state.last_training_run.update({
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "status": "FAILED",
            "error": str(e),
            "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
        })
        model_state.current_training_job = None

def train_als_model_with_warmstart(
    interactions: List[InteractionData], 
    old_model: AlternatingLeastSquares
) -> AlternatingLeastSquares:
    """Train with warm-start from existing model"""
    logger.info("🔥 Training ALS with WARM-START...")
    
    interactions_df = pd.DataFrame([
        {'user_id': i.user_id, 'product_id': i.product_id, 'weight': i.weight}
        for i in interactions
    ])
    
    logger.info(f"Training data: {len(interactions_df)} interactions")
    
    # Create new model with same hyperparameters
    new_model = AlternatingLeastSquares(
        n_factors=old_model.n_factors,
        reg_param=old_model.reg_param,
        n_iterations=5,  # Fewer iterations for warm-start
        alpha=old_model.alpha
    )
    
    # Fit
    new_model.fit(interactions_df)
    
    # Transfer knowledge from old model (70% old + 30% new)
    for user_id, user_idx in new_model.user_map.items():
        if user_id in old_model.user_map:
            old_idx = old_model.user_map[user_id]
            new_model.user_factors[user_idx] = (
                0.7 * old_model.user_factors[old_idx] +
                0.3 * new_model.user_factors[user_idx]
            )
    
    for item_id, item_idx in new_model.item_map.items():
        if item_id in old_model.item_map:
            old_idx = old_model.item_map[item_id]
            new_model.item_factors[item_idx] = (
                0.7 * old_model.item_factors[old_idx] +
                0.3 * new_model.item_factors[item_idx]
            )
    
    logger.info(f"✅ Trained: {len(new_model.user_map)} users, {len(new_model.item_map)} items")
    return new_model

# Tiếp tục từ dòng cuối cùng của file hiện tại...

def train_als_model(interactions: List[InteractionData]) -> AlternatingLeastSquares:
    """Train ALS model from scratch"""
    logger.info("🔥 Training ALS model (FULL RETRAIN)...")
    
    interactions_df = pd.DataFrame([
        {'user_id': i.user_id, 'product_id': i.product_id, 'weight': i.weight}
        for i in interactions
    ])
    
    logger.info(f"Training data: {len(interactions_df)} interactions")
    
    als = AlternatingLeastSquares(
        n_factors=config.ALS_N_FACTORS,
        reg_param=config.ALS_REGULARIZATION,
        n_iterations=config.ALS_N_ITERATIONS,
        alpha=config.ALS_ALPHA
    )
    
    als.fit(interactions_df)
    
    logger.info(f"✅ Trained: {len(als.user_map)} users, {len(als.item_map)} items")
    return als


def generate_phobert_embeddings(products: List[ProductData]) -> tuple[np.ndarray, List[str]]:
    """Generate PhoBERT embeddings (placeholder with random)"""
    logger.info("🤖 Generating PhoBERT embeddings...")
    
    n_products = len(products)
    embedding_dim = 768
    
    # TODO: Replace with actual PhoBERT inference
    embeddings = np.random.randn(n_products, embedding_dim).astype(np.float32)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    product_ids = [p.product_id for p in products]
    
    logger.info(f"✅ Generated embeddings: {embeddings.shape}")
    logger.warning("⚠️  Using random embeddings - implement actual PhoBERT!")
    
    return embeddings, product_ids


# ============================================
# STARTUP & SHUTDOWN
# ============================================
@app.on_event("startup")
async def startup():
    """Initialize service"""
    try:
        logger.info("=" * 60)
        logger.info("Starting Recommend Service...")
        logger.info("=" * 60)

        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        # Register with Eureka
        await eureka_client.init_async(
            eureka_server=config.EUREKA_SERVER,
            app_name=config.SERVICE_NAME,
            instance_port=config.SERVICE_PORT,
            instance_ip=ip_address,
            instance_host=hostname,
            health_check_url=f"http://{ip_address}:{config.SERVICE_PORT}/health",
            status_page_url=f"http://{ip_address}:{config.SERVICE_PORT}/admin/metrics",
            renewal_interval_in_secs=30,
            duration_in_secs=90,
        )

        logger.info("✅ Registered with Eureka")

        # Create directories
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load ALS model
        try:
            if config.ALS_MODEL_PATH.exists():
                model_state.als_model = AlternatingLeastSquares.load(str(config.ALS_MODEL_PATH))
                logger.info(f"✅ ALS loaded: {len(model_state.als_model.user_map)} users")
            else:
                logger.warning("⚠️  ALS model not found")
        except Exception as e:
            logger.error(f"❌ Failed to load ALS: {e}")

        # Load PhoBERT
        try:
            if config.PHOBERT_EMBEDDINGS_PATH.exists():
                with open(config.PHOBERT_EMBEDDINGS_PATH, 'rb') as f:
                    data = pickle.load(f)
                model_state.phobert_embeddings = data['embeddings']
                model_state.product_ids = data['product_ids']
                logger.info(f"✅ PhoBERT loaded: {len(model_state.product_ids)} products")
            else:
                logger.warning("⚠️  PhoBERT not found")
        except Exception as e:
            logger.error(f"❌ Failed to load PhoBERT: {e}")

        model_state.models_loaded = bool(model_state.als_model or model_state.phobert_embeddings)
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Startup error: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown"""
    try:
        logger.info("Shutting down...")
        await eureka_client.stop_async()
        logger.info("✅ Deregistered from Eureka")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")



# HELPER FUNCTIONS

def analyze_dataset_size(
    base_interactions: List[InteractionData],
    accumulated_interactions: List[InteractionData],
    api_interactions: List[InteractionData]
) -> Dict[str, Any]:
    """
    Phân tích kích thước dataset để quyết định training strategy
    
    Returns:
        {
            'total_size': int,
            'api_size': int,
            'accumulated_size': int,
            'base_size': int,
            'recommended_strategy': str,  # 'full_retrain' | 'warmstart' | 'incremental'
            'reason': str
        }
    """
    base_size = len(base_interactions)
    accumulated_size = len(accumulated_interactions)
    api_size = len(api_interactions)
    total_size = base_size + accumulated_size + api_size
    
    # Decision logic
    if total_size < config.INCREMENTAL_THRESHOLD:
        # Dataset nhỏ → Full retrain nhanh
        strategy = 'full_retrain'
        reason = f"Total {total_size:,} < threshold {config.INCREMENTAL_THRESHOLD:,}. Full retrain nhanh và đảm bảo quality."
    
    elif config.INCREMENTAL_THRESHOLD <= total_size < config.FULL_RETRAIN_THRESHOLD:
        # Dataset trung bình → Warmstart
        strategy = 'warmstart'
        reason = f"Total {total_size:,} trong khoảng [{config.INCREMENTAL_THRESHOLD:,}, {config.FULL_RETRAIN_THRESHOLD:,}). Dùng warmstart để tiết kiệm thời gian."
    
    else:
        # Dataset lớn → Incremental
        strategy = 'incremental'
        reason = f"Total {total_size:,} > {config.FULL_RETRAIN_THRESHOLD:,}. Chuyển sang incremental learning."
    
    # Special case: API data quá ít
    if api_size < 100:
        strategy = 'skip'
        reason = f"API data chỉ {api_size} interactions (< 100). Không đủ để update model, giữ nguyên model hiện tại."
    
    return {
        'total_size': total_size,
        'api_size': api_size,
        'accumulated_size': accumulated_size,
        'base_size': base_size,
        'recommended_strategy': strategy,
        'reason': reason
    }
def get_collaborative_recommendations(user_id: str, top_k: int = 10) -> Optional[List[str]]:
    """Get personalized recommendations"""
    if model_state.als_model is None or str(user_id) not in model_state.als_model.user_map:
        return None
    
    try:
        return model_state.als_model.predict_for_user(user_id, top_k=top_k)
    except Exception as e:
        logger.warning(f"Collaborative failed: {e}")
        return None


def get_popular_fallback(top_k: int = 10) -> List[str]:
    """Fallback to popular items"""
    if model_state.product_ids:
        return model_state.product_ids[:top_k]
    return []
# ============================================
# ADDITIONAL HELPER FUNCTIONS FOR PRODUCT DETAIL
# ============================================
def get_content_recommendations_phobert(product_id: str, top_k: int = 10) -> Optional[List[str]]:
    """
    Content-based recommendations using PhoBERT embeddings
    
    Args:
        product_id: Target product ID
        top_k: Number of recommendations
        
    Returns:
        List of similar product IDs or None if unavailable
    """
    if model_state.phobert_embeddings is None or model_state.product_ids is None:
        logger.warning("PhoBERT embeddings not available")
        return None
    
    if str(product_id) not in model_state.product_ids:
        logger.warning(f"Product {product_id} not found in PhoBERT embeddings")
        return None
    
    try:
        # Get product index
        prod_idx = model_state.product_ids.index(str(product_id))
        target_vec = model_state.phobert_embeddings[prod_idx]
        
        # Calculate cosine similarity with all products
        similarities = np.dot(model_state.phobert_embeddings, target_vec)
        
        # Get top-k similar products (exclude self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        recommendations = [model_state.product_ids[i] for i in top_indices]
        logger.info(f"PhoBERT content-based: {product_id} -> {len(recommendations)} recommendations")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"PhoBERT content-based failed for {product_id}: {e}")
        return None


def get_similar_items_als(product_id: str, top_k: int = 10) -> Optional[List[str]]:
    """
    Item-item similarity using ALS collaborative filtering
    
    Args:
        product_id: Target product ID
        top_k: Number of similar items
        
    Returns:
        List of similar product IDs or None if unavailable
    """
    if model_state.als_model is None:
        logger.warning("ALS model not available")
        return None
    
    if str(product_id) not in model_state.als_model.item_map:
        logger.warning(f"Product {product_id} not found in ALS model")
        return None
    
    try:
        similar_items = model_state.als_model.get_similar_items(str(product_id), top_k=top_k)
        
        if similar_items:
            logger.info(f"ALS item-similarity: {product_id} -> {len(similar_items)} items")
            return similar_items
        else:
            logger.warning(f"No similar items found for {product_id}")
            return None
            
    except Exception as e:
        logger.error(f"ALS item similarity failed for {product_id}: {e}")
        return None


def hybrid_merge_recommendations(
    content_recs: Optional[List[str]], 
    collab_recs: Optional[List[str]],
    content_weight: float = 0.6, 
    top_k: int = 10
) -> List[str]:
    """
    Merge content-based and collaborative recommendations
    
    Strategy: 
    - If both available: 60% content + 40% collaborative (default)
    - If only content: 100% content
    - If only collaborative: 100% collaborative
    - If neither: empty list
    
    Args:
        content_recs: Content-based recommendations
        collab_recs: Collaborative recommendations
        content_weight: Weight for content-based (0-1)
        top_k: Number of final recommendations
        
    Returns:
        Merged list of product IDs
    """
    # Case 1: Both available - hybrid merge
    if content_recs and collab_recs:
        num_content = int(top_k * content_weight)
        num_collab = top_k - num_content
        
        result = []
        seen = set()
        
        # Add from content-based
        for pid in content_recs[:num_content]:
            if pid not in seen:
                result.append(pid)
                seen.add(pid)
        
        # Add from collaborative
        for pid in collab_recs[:num_collab]:
            if pid not in seen and len(result) < top_k:
                result.append(pid)
                seen.add(pid)
        
        # Fill remaining from content if needed
        for pid in content_recs[num_content:]:
            if pid not in seen and len(result) < top_k:
                result.append(pid)
                seen.add(pid)
        
        logger.info(f"Hybrid merge: {len(result)} items (content + collaborative)")
        return result[:top_k]
    
    # Case 2: Only content available
    elif content_recs:
        logger.info(f"Using 100% content-based: {len(content_recs[:top_k])} items")
        return content_recs[:top_k]
    
    # Case 3: Only collaborative available
    elif collab_recs:
        logger.info(f"Using 100% collaborative: {len(collab_recs[:top_k])} items")
        return collab_recs[:top_k]
    
    # Case 4: Neither available
    else:
        logger.warning("No recommendations available from either strategy")
        return []


# API ENDPOINTS
@app.get("/api/v1/internal/recommend/product-detail/{product_id}")
async def recommend_product_detail(
    product_id: str,
    user_id: Optional[int] = Header(None, alias="X-User-Id"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """
    Product detail page recommendations
    
    Strategy:
    - Guest users: Content-based only (PhoBERT similarity)
    - Logged-in users: Hybrid (Content-based + Item-item collaborative)
    
    Args:
        product_id: Target product ID
        user_id: User ID from header (optional)
        top_k: Number of recommendations
        
    Returns:
        JSON with product_ids, strategy, and count
        
    Raises:
        404: Product not found in recommendation system
        503: Models not loaded
    """
    logger.info(f"Product detail request: product_id={product_id}, user_id={user_id}, top_k={top_k}")
    
    # Validate user_id type
    if user_id is not None and not isinstance(user_id, int):
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid user_id format: {user_id}, treating as guest")
            user_id = None
    
    # Check if models are loaded
    if not model_state.models_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded yet. Please wait for training to complete."
        )
    
    # Always try to get content-based recommendations (PhoBERT)
    content_recs = get_content_recommendations_phobert(product_id, top_k=top_k)
    
    # Check if product exists in at least one model
    product_in_phobert = content_recs is not None
    product_in_als = (
        model_state.als_model is not None and 
        str(product_id) in model_state.als_model.item_map
    )
    
    if not product_in_phobert and not product_in_als:
        raise HTTPException(
            status_code=404,
            detail=f"Product {product_id} not found in recommendation system. "
                   f"Product may not exist or hasn't been included in training data yet."
        )
    
    # GUEST USER: Content-based only
    if not user_id:
        if content_recs and len(content_recs) > 0:
            return {
                "product_ids": content_recs,
                "strategy": "content-based",
                "count": len(content_recs),
                "source_product_id": product_id
            }
        else:
            # Fallback to popular if content-based fails
            popular = get_popular_fallback(top_k)
            return {
                "product_ids": popular,
                "strategy": "popular_fallback",
                "count": len(popular),
                "source_product_id": product_id
            }
    
    # LOGGED-IN USER: Try hybrid strategy
    try:
        # Get collaborative item-item recommendations
        collab_recs = get_similar_items_als(product_id, top_k=top_k)
        
        # Merge content + collaborative
        if content_recs or collab_recs:
            hybrid_recs = hybrid_merge_recommendations(
                content_recs=content_recs,
                collab_recs=collab_recs,
                content_weight=0.6,  # 60% content, 40% collaborative
                top_k=top_k
            )
            
            if len(hybrid_recs) > 0:
                strategy = "hybrid" if (content_recs and collab_recs) else (
                    "content-based" if content_recs else "collaborative"
                )
                
                return {
                    "product_ids": hybrid_recs,
                    "strategy": strategy,
                    "count": len(hybrid_recs),
                    "source_product_id": product_id
                }
        
    except Exception as e:
        logger.error(f"Hybrid strategy failed: {e}")
    
    # Final fallback to popular
    popular = get_popular_fallback(top_k)
    return {
        "product_ids": popular,
        "strategy": "popular_fallback",
        "count": len(popular),
        "source_product_id": product_id
    }
@app.get("/api/v1/internal/recommend/homepage")
async def recommend_homepage(
    user_id: Optional[int] = Header(None, alias="X-User-Id"),
    top_k: int = Query(10, ge=1, le=50)
):
    """Homepage recommendations"""
    logger.info(f"Homepage: user_id={user_id}, top_k={top_k}")
    
    if not model_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if user_id:
        recs = get_collaborative_recommendations(str(user_id), top_k)
        if recs and len(recs) > 0:
            return {"product_ids": recs, "strategy": "personalized", "count": len(recs)}

    popular = get_popular_fallback(top_k)
    return {"product_ids": popular, "strategy": "popular", "count": len(popular)}


@app.post("/api/v1/internal/recommend/train", status_code=202)
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Trigger training"""
    if model_state.current_training_job:
        raise HTTPException(status_code=409, detail="Training already running")

    job_id = str(uuid.uuid4())
    model_state.current_training_job = job_id

    model_state.last_training_run = {
        "job_id": job_id,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "status": "RUNNING"
    }

    background_tasks.add_task(run_training_pipeline, job_id, request)

    return {
        "status": "Training started",
        "job_id": job_id,
        "message": "Check /admin/metrics for progress"
    }


@app.get("/admin/metrics")
async def get_metrics():
    """Service metrics với thông tin strategy"""
    return {
        "service_status": "ONLINE" if model_state.models_loaded else "DEGRADED",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "last_training_run": model_state.last_training_run,
        "current_training_job": model_state.current_training_job,
        "training_thresholds": {  # 🆕
            "incremental_threshold": config.INCREMENTAL_THRESHOLD,
            "full_retrain_threshold": config.FULL_RETRAIN_THRESHOLD,
            "current_strategy": model_state.last_training_run.get('strategy', 'N/A')
        },
        "active_models": {
            "als_model": {
                "status": "loaded" if model_state.als_model else "unavailable",
                "users": len(model_state.als_model.user_map) if model_state.als_model else 0,
                "items": len(model_state.als_model.item_map) if model_state.als_model else 0
            },
            "phobert_model": {
                "status": "loaded" if model_state.phobert_embeddings is not None else "unavailable",
                "total_items": len(model_state.product_ids) if model_state.product_ids else 0
            }
        }
    }


@app.get("/health")
async def health_check():
    """Health check"""
    if not model_state.models_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "UNHEALTHY", "error": "No models loaded"}
        )

    return {
        "status": "HEALTHY",
        "models_loaded": ["als" if model_state.als_model else None, 
                         "phobert" if model_state.phobert_embeddings else None],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/")
async def root():
    """Service info"""
    return {
        "service": "Recommend Service",
        "version": "1.0.0",
        "status": "online" if model_state.models_loaded else "initializing"
    }

# MAIN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.SERVICE_PORT)
