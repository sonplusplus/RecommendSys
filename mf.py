import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pickle
from load_data import load_tf_data


class AlternatingLeastSquares:
    """
    click=1, add_to_cart=2, purchase=3
    """
    
    def __init__(self, n_factors=50, n_iterations=15, reg_param=0.01, alpha=40):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg_param = reg_param          #loss func
        self.alpha = alpha                  #confidence
        
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
    
    def fit(self, interactions_df):
        required_cols = ['user_id', 'product_id', 'weight']
        missing = [col for col in required_cols if col not in interactions_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_map.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Create WEIGHTED user-item interaction matrix
        rows = interactions_df['user_id'].map(self.user_map)
        cols = interactions_df['product_id'].map(self.item_map)
        
        
        weights = interactions_df['weight'].values
        
        
        P = csr_matrix((np.ones(len(interactions_df)), (rows, cols)), shape=(n_users, n_items))
        
        # Confidence matrix
        # C_ui = 1 + alpha * r_ui (rui: weight)
        confidence_data = 1 + self.alpha * weights
        C = csr_matrix((confidence_data, (rows, cols)), shape=(n_users, n_items))
        
        print(f"\n{'='*60}")
        print(f"Weighted ALS Training")
        print(f"{'='*60}")
        print(f"Matrix: {n_users:,} users × {n_items:,} items")
        print(f"Total interactions: {len(interactions_df):,}")
        print(f"Sparsity: {100 * (1 - len(interactions_df) / (n_users * n_items)):.2f}%")
        print(f"\nWeight distribution:")
        print(interactions_df['weight'].value_counts().sort_index())
        print(f"\nConfidence range: {confidence_data.min():.2f} - {confidence_data.max():.2f}")
        print(f"Alpha parameter: {self.alpha}")
        
        # Initialize factors
        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.01
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.01
        
        # ALS iterations
        print(f"\nTraining for {self.n_iterations} iterations...")
        
        for iteration in range(self.n_iterations):
            # cố định item thay đổi user
            self.user_factors = self._solve_weighted(P, C, self.item_factors, self.reg_param)
            
            # cố định user, thay đổi item
            self.item_factors = self._solve_weighted(P.T, C.T, self.user_factors, self.reg_param)
            
            if iteration % 5 == 0 or iteration == self.n_iterations - 1:
                loss = self._compute_weighted_loss(P, C)
                print(f"Iteration {iteration+1}/{self.n_iterations}, Loss: {loss:.4f}")
        
        print(f"  - User embeddings shape: {self.user_factors.shape}")
        print(f"  - Item embeddings shape: {self.item_factors.shape}")
        
        return self
    
    def _solve_weighted(self, P, C, fixed_factors, reg):
        """
        formula: (Y^T C Y + λI) x = Y^T C P
        """
        n = P.shape[0]
        k = fixed_factors.shape[1]
        factors = np.zeros((n, k))
        
        YtY = fixed_factors.T @ fixed_factors
        reg_eye = reg * np.eye(k)
        
        for u in range(n):
            # Get user u's data
            p_u = P[u].toarray().flatten()  
            c_u = C[u].toarray().flatten()  
            
            # Find items with interactions
            idx = p_u > 0
            
            if idx.sum() == 0:
                continue
            
            # Y_u: item factors for interacted items
            Y_u = fixed_factors[idx]
            c_u_nonzero = c_u[idx]
            p_u_nonzero = p_u[idx]
            
            # C_u as diagonal matrix (only non-zero part)
            C_u_diag = np.diag(c_u_nonzero)
            
            # Solve: (Y_u^T C_u Y_u + λI) x = Y_u^T C_u p_u
            A = Y_u.T @ C_u_diag @ Y_u + reg_eye
            b = Y_u.T @ C_u_diag @ p_u_nonzero
            
            factors[u] = np.linalg.solve(A, b)
        
        return factors
    
    def _compute_weighted_loss(self, P, C):
        """Calculate weighted reconstruction error"""
        predictions = self.user_factors @ self.item_factors.T
        P_dense = P.toarray()
        C_dense = C.toarray()
        
        # Weighted squared error: sum(C_ui * (P_ui - X_ui)^2)
        diff = P_dense - predictions
        weighted_error = np.sum(C_dense * (diff ** 2))
        
        # Regularization
        reg_loss = self.reg_param * (
            np.sum(self.user_factors ** 2) + 
            np.sum(self.item_factors ** 2)
        )
        
        return weighted_error / P.nnz + reg_loss
    
    def predict_for_user(self, user_id, top_k=10):
        """
        Personalized recommendations for a user
        Returns top-k product IDs
        """
        if user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        scores = self.user_factors[user_idx] @ self.item_factors.T
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [self.reverse_item_map[idx] for idx in top_indices]
    
    def predict_for_user_with_scores(self, user_id, top_k=10):
        """
        Returns recommendations with confidence scores
        """
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
        """
        Find similar items based on item factors
        Logic: Items with similar embeddings
        """
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
        
        # Top-k (exclude itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [self.reverse_item_map[idx] for idx in top_indices]
    
    def save(self, path="models/als_model.pkl"):
        """Save model to disk"""
        model_data = {
            'user_factors': self.user_factors,      
            'item_factors': self.item_factors,      
            'user_map': self.user_map,              
            'item_map': self.item_map,              
            'reverse_user_map': self.reverse_user_map,  
            'reverse_item_map': self.reverse_item_map,  
            'n_factors': self.n_factors,
            'n_iterations': self.n_iterations,
            'reg_param': self.reg_param,
            'alpha': self.alpha  
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {path}")
        import os
        file_size = os.path.getsize(path) / 1e6
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Users: {len(self.user_map):,}")
        print(f"  Items: {len(self.item_map):,}")
    
    @classmethod
    def load(cls, path="models/als_model.pkl"):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            n_factors=model_data['n_factors'],
            n_iterations=model_data['n_iterations'],
            reg_param=model_data['reg_param'],
            alpha=model_data.get('alpha', 40)  
        )
        
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.user_map = model_data['user_map']
        model.item_map = model_data['item_map']
        model.reverse_user_map = model_data['reverse_user_map']
        model.reverse_item_map = model_data['reverse_item_map']
        
        print(f"Model loaded from {path}")
        print(f"  - Users: {len(model.user_map):,}")
        print(f"  - Items: {len(model.item_map):,}")
        print(f"  - Factors: {model.n_factors}")
        print(f"  - Alpha: {model.alpha}")
        
        return model


def evaluate_item_similarity(model, test_df, k=10):
    """
    Evaluate item-to-item similarity
    Check if similar items co-occur in test set
    """
    print(f"\nEvaluating item-to-item similarity...")
    
    # Get popular items
    item_counts = test_df['product_id'].value_counts()
    popular_items = item_counts[item_counts >= 3].index[:50]
    
    hits = 0
    total = 0
    
    for item_id in popular_items:
        if item_id not in model.item_map:
            continue
        
        users_bought_this = set(test_df[test_df['product_id'] == item_id]['user_id'])
        similar_items = model.get_similar_items(item_id, top_k=k)
        
        if similar_items is None:
            continue
        
        for sim_item in similar_items:
            users_bought_similar = set(test_df[test_df['product_id'] == sim_item]['user_id'])
            if len(users_bought_this & users_bought_similar) > 0:
                hits += 1
                break
        
        total += 1
    
    accuracy = hits / total if total > 0 else 0
    
    print(f"\nResults:")
    print(f"  - Co-occurrence rate: {accuracy*100:.2f}%")
    print(f"  - Hit rate: {hits}/{total} items")
    
    return accuracy


def train_als_model():
    """Train weighted ALS model"""
    
    # Load data
    print("\n1. Loading data with weights...")
    dataset, unique_user_ids, unique_product_ids, _, _, _, _, weights = load_tf_data()
    
    if dataset is None:
        print("Failed load data")
        return
    

    interactions = []
    for batch in dataset.batch(1000):
        for i in range(len(batch['user_id'])):
            interactions.append({
                'user_id': batch['user_id'][i].numpy().decode('utf-8'),
                'product_id': batch['product_id'][i].numpy().decode('utf-8'),
                'weight': float(batch['weight'][i].numpy())
            })
    
    df = pd.DataFrame(interactions)
    

    print(f"Dataset Statistics")
    print(f"Total interactions: {len(df):,}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
    print(f"Avg interactions/user: {len(df)/df['user_id'].nunique():.1f}")
    print(f"Avg interactions/product: {len(df)/df['product_id'].nunique():.1f}")
    print(f"\nWeight distribution:")
    print(df['weight'].value_counts().sort_index())
    
    # Split data
    print("\n2. Splitting data (80/20)...")
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['weight'])
    print(f"Train: {len(train_df):,} interactions")
    print(f"Test:  {len(test_df):,} interactions")
    
    # Train ALS
    print("\n3. Training Weighted ALS model...")
    
    als_model = AlternatingLeastSquares(
        n_factors=50,
        n_iterations=15,
        reg_param=0.01,
        alpha=40  
    )
    
    als_model.fit(train_df)
    
    # Evaluate
    print("\n4. Evaluation")
    evaluate_item_similarity(als_model, test_df, k=10)
    
    # Save model
    print("\n5. Saving model...")
    als_model.save("models/als_model.pkl")
    
    # Demo
    print("\n6. Demo Recommendations")
    print("="*60)
    
    # Demo 1: User personalization
    sample_user = df['user_id'].value_counts().head(1).index[0]
    print(f"\nPersonalized recommendations for user '{sample_user}':")
    user_recs = als_model.predict_for_user_with_scores(sample_user, top_k=5)
    if user_recs:
        for i, rec in enumerate(user_recs, 1):
            print(f"  {i}. Product {rec['product_id']} (confidence: {rec['confidence']:.3f})")
    
    # Demo 2: Item similarity
    sample_items = df['product_id'].value_counts().head(3).index
    print(f"\nItem-to-Item Similarity:")
    for sample_item in sample_items:
        similar = als_model.get_similar_items(sample_item, top_k=5)
        if similar:
            print(f"\nSimilar items to '{sample_item}':")
            for i, product_id in enumerate(similar, 1):
                print(f"  {i}. {product_id}")
    
    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_als_model()