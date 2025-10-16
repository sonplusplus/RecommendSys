import numpy as np


class ALSModel:   
    def __init__(self, data):
        self.item_factors = data['item_factors']
        self.item_map = data['item_map']
        self.reverse_item_map = data['reverse_item_map']
        self.n_factors = data['n_factors']
        
    # When need to display personalized recommendations on homepage
    def predict_for_user(self, user_id, top_k=10):
        """
        Gợi ý cá nhân hóa dựa trên preferences của user
        Logic: User vector · Item vectors = scores
        → Sản phẩm có score cao = phù hợp với sở thích user
        """
        if user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        scores = self.user_factors[user_idx] @ self.item_factors.T
        top_indices = np.argsort(scores)[::-1][:top_k]
        rcm = [self.reverse_item_map[idx] for idx in top_indices]
        return rcm
    
    def predict_for_user_with_scores(self, user_id, top_k=10):
        """
        Trả về recommendations kèm confidence scores
        """
        if user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        scores = self.user_factors[user_idx] @ self.item_factors.T
        
        # Normalize scores về [0, 1]
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
        """
        Tìm sản phẩm tương tự dựa trên collaborative patterns
        Logic: "Users who bought A also bought B"
        """
        if item_id not in self.item_map:
            return None
        
        item_idx = self.item_map[item_id]
        item_vec = self.item_factors[item_idx]
        
        # Cosine similarity giữa item vectors
        item_norm = item_vec / (np.linalg.norm(item_vec) + 1e-8)
        all_norms = self.item_factors / (
            np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = all_norms @ item_norm
        
        # Top-k (bỏ chính nó)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [self.reverse_item_map[idx] for idx in top_indices]