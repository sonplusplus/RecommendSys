import numpy as np
import pickle


class ALSModel:

    
    def __init__(self, model_data):
        # User factors 
        self.user_factors = model_data.get('user_factors')
        self.user_map = model_data.get('user_map', {})
        self.reverse_user_map = model_data.get('reverse_user_map', {})
        
        # Item factors 
        self.item_factors = model_data['item_factors']
        self.item_map = model_data['item_map']
        self.reverse_item_map = model_data['reverse_item_map']
        
        # Model params
        self.n_factors = model_data['n_factors']
        self.alpha = model_data.get('alpha', 40)
    
    def has_user_personalization(self):
        return self.user_factors is not None and len(self.user_map) > 0
    
    def predict_for_user(self, user_id, top_k=10):
        """score = user_vector · item_vectors"""
        if self.user_factors is None or user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        scores = self.user_factors[user_idx] @ self.item_factors.T
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [self.reverse_item_map[idx] for idx in top_indices]
    
    def predict_for_user_with_scores(self, user_id, top_k=10):
        """Get recommendations with normalized confidence scores [0,1]"""
        if self.user_factors is None or user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        scores = self.user_factors[user_idx] @ self.item_factors.T
        
        # Normalize to [0, 1]
        min_score, max_score = scores.min(), scores.max()
        normalized_scores = (scores - min_score) / (max_score - min_score + 1e-8)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            {
                'product_id': self.reverse_item_map[idx],
                'score': float(scores[idx]),
                'confidence': float(normalized_scores[idx])
            }
            for idx in top_indices
        ]
    
    def get_similar_items(self, item_id, top_k=10):
        """Find similar items via cosine similarity on item factors"""
        if item_id not in self.item_map:
            return None
        
        item_idx = self.item_map[item_id]
        item_vec = self.item_factors[item_idx]

        item_norm = item_vec / (np.linalg.norm(item_vec) + 1e-8)    #avoid divde 0
        all_norms = self.item_factors / (
            np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8
        )
        similarities = all_norms @ item_norm
        
        # topk sp
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        return [self.reverse_item_map[idx] for idx in top_indices]
    
    def get_similar_items_with_scores(self, item_id, top_k=10):
        """Similar items with similarity scores"""
        if item_id not in self.item_map:
            return None
        
        item_idx = self.item_map[item_id]
        item_vec = self.item_factors[item_idx]
        
        item_norm = item_vec / (np.linalg.norm(item_vec) + 1e-8)
        all_norms = self.item_factors / (
            np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8
        )
        similarities = all_norms @ item_norm
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [
            {
                'product_id': self.reverse_item_map[idx],
                'similarity': float(similarities[idx])
            }
            for idx in top_indices
        ]
    
    @classmethod
    def load_from_file(cls, path="models/als_model.pkl"):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        return cls(model_data)