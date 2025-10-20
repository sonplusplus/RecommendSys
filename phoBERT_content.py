import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm

# Config
PRODUCTS_CSV = "data/products.csv"
OUTPUT_PATH = "data/phobert_embeddings.pkl"
MODEL_NAME = "vinai/phobert-base"
BATCH_SIZE = 16  

class PhoBERTContentModel:
    def __init__(self, model_name=MODEL_NAME):
        """Initialize PhoBERT model"""
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def encode_text(self, text):
        """Encode single text to embedding"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy()
    
    def encode_batch(self, texts, batch_size=BATCH_SIZE):
        """Encode multiple texts with batching"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            

            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
            

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return np.vstack(embeddings)

def build_vietnamese_content_model():
    """Build content_based model using PhoBERT"""
    
    # Load products
    df = pd.read_csv(PRODUCTS_CSV)
    df['product_id'] = df['product_id'].astype(str)
    
    df['name'] = df.get('name', df.get('product_name', '')).fillna('')
    df['category'] = df['category'].fillna('')
    df['brand'] = df['brand'].fillna('')
    df['description'] = df['description'].fillna('')
    

    df['combined_features'] = (
        df['name'] + '. ' +
        df['name'] + '. ' +  
        df['category'] + '. ' +
        df['brand'] + '. ' +
        df['description']
    )
    
    print(f"\nTotal products: {len(df)}")
    print(f"Sample combined text:\n{df['combined_features'].iloc[0]}\n")
    
    # Init PhoBERT
    phobert = PhoBERTContentModel()
    
    embeddings = phobert.encode_batch(df['combined_features'].tolist())


    product_ids = df['product_id'].tolist()
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
 
    print("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # tránh divide 0
    
    # cosine similarity
    print("\nTesting similarity calculation...")
    similarities = cosine_similarity(embeddings)
    
    test_idx = 0
    top_5_idx = np.argsort(similarities[test_idx])[::-1][1:6]
    
    print(f"\nTop 5 similar products to {product_ids[test_idx]}:")
    print(f"  Original: {df['combined_features'].iloc[test_idx][:100]}...")
    print("\nSimilar products:")
    for i, idx in enumerate(top_5_idx, 1):
        sim = similarities[test_idx, idx]
        print(f"  {i}. {product_ids[idx]} (sim: {sim:.4f})")
        print(f"     {df['combined_features'].iloc[idx][:100]}...")
    
    # Save với metadata
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,  # Đã normalized
            'product_ids': product_ids,
            'model_name': MODEL_NAME,
            'normalized': True,
            'embedding_dim': embeddings.shape[1]
        }, f)
    
    print("Vietnamese content model saved!")
    print(f"{len(product_ids)} products encoded")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    
    return embeddings, product_ids

def get_recommendations(product_id, embeddings, product_ids, top_k=10):
    """Get content-based recommendations"""
    
    if product_id not in product_ids:
        raise ValueError(f"Product {product_id} not found")
    
    prod_idx = product_ids.index(product_id)
    similarities = np.dot(embeddings, embeddings[prod_idx])
    
    #top k sp 
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    recommendations = [
        {
            'product_id': product_ids[i],
            'similarity': float(similarities[i])
        }
        for i in top_indices
    ]
    
    return recommendations

def load_embeddings(path=OUTPUT_PATH):

    print(f"Loading embeddings from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data['product_ids'])} products")
    print(f"Model: {data['model_name']}")
    print(f"Normalized: {data.get('normalized', False)}")
    
    return data['embeddings'], data['product_ids']

if __name__ == "__main__":
    import os
    
    print("Building PhoBERT embeddings...")
    embeddings, product_ids = build_vietnamese_content_model()
    

    if embeddings is not None and len(product_ids) > 0:
        test_product = product_ids[0]
        recs = get_recommendations(test_product, embeddings, product_ids, top_k=5)
        
        print(f"\nTest successful!")
        print(f"Recommendations for {test_product}:")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec['product_id']} (similarity: {rec['similarity']:.4f})")
        