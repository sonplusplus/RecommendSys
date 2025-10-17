import numpy as np


class ALSModel:
    
    def __init__(self, data):
        # Get top-k (exclude itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = [
            {
                'product_id': self.reverse_item_map[idx],
                'similarity': float(similarities[idx])
            }
            for idx in top_indices
        ]
        
        return results
    
    def has_user_personalization(self):
        """Check if model supports user personalization"""
        return self.user_factors is not None and len(self.user_map) > 0 
        self.user_factors = data.get('user_factors')
        self.user_map = data.get('user_map', {})
        self.reverse_user_map = data.get('reverse_user_map', {})
        
        # Load item factors
        self.item_factors = data['item_factors']
        self.item_map = data['item_map']
        self.reverse_item_map = data['reverse_item_map']
        
        # Model params
        self.n_factors = data['n_factors']
        self.alpha = data.get('alpha', 40)
    
    def predict_for_user(self, user_id, top_k=10):
        
        if not self.user_factors is not None:
            return None
        
        if user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        
        # user_vector · item_vectors
        scores = self.user_factors[user_idx] @ self.item_factors.T
        
        # Get top-k items
        top_indices = np.argsort(scores)[::-1][:top_k]
        rcm = [self.reverse_item_map[idx] for idx in top_indices]
        
        return rcm
    
    def predict_for_user_with_scores(self, user_id, top_k=10):
        
        if not self.user_factors is not None:
            return None
            
        if user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        scores = self.user_factors[user_idx] @ self.item_factors.T
        
        # Normalize scores to [0, 1]
        min_score = scores.min()
        max_score = scores.max()
        normalized_scores = (scores - min_score) / (max_score - min_score + 1e-8)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            {
                'product_id': self.reverse_item_map[idx],
                'score': float(scores[idx]),
                'confidence': float(normalized_scores[idx])
            }
            for idx in top_indices
        ]
        
        return results
    
    def get_similar_items(self, item_id, top_k=10):
       
        if item_id not in self.item_map:
            return None
        
        item_idx = self.item_map[item_id]
        item_vec = self.item_factors[item_idx]
        
        # Cosine similarity between item vectors
        item_norm = item_vec / (np.linalg.norm(item_vec) + 1e-8)
        all_norms = self.item_factors / (
            np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = all_norms @ item_norm
        
        # top-k sp
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [self.reverse_item_map[idx] for idx in top_indices]
    
    def get_similar_items_with_scores(self, item_id, top_k=10):
        if item_id not in self.item_map:
            return None
        
        item_idx = self.item_map[item_id]
        item_vec = self.item_factors[item_idx]
        
        # Cosine similarity
        item_norm = item_vec / (np.linalg.norm(item_vec) + 1e-8)
        all_norms = self.item_factors / (
            np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = all_norms @ item_norm
        
        