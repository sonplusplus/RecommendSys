import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pickle
from load_data import load_tf_data


class AlternatingLeastSquares:
    
    def __init__(self, n_factors=50, n_iterations=15, reg_param=0.01):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg_param = reg_param
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
    
    def fit(self, interactions_df):
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_map.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Create user-item interaction matrix
        rows = interactions_df['user_id'].map(self.user_map)
        cols = interactions_df['product_id'].map(self.item_map)
        data = np.ones(len(interactions_df))
        
        R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        print(f"Matrix: {n_users} users × {n_items} items")
        print(f"Sparsity: {100 * (1 - R.nnz / (n_users * n_items)):.2f}%")
        print(f"Total interactions: {R.nnz}")
        
        # Initialize factors
        self.user_factors = np.random.randn(n_users, self.n_factors) * 0.01
        self.item_factors = np.random.randn(n_items, self.n_factors) * 0.01
        
        # ALS iterations
        print(f"\nTraining for {self.n_iterations} iterations...")
        
        for iteration in range(self.n_iterations):
            # Fix items, solve for users
            self.user_factors = self._solve_factors(R, self.item_factors, self.reg_param)
            
            # Fix users, solve for items
            self.item_factors = self._solve_factors(R.T, self.user_factors, self.reg_param)
            
            if iteration % 5 == 0 or iteration == self.n_iterations - 1:
                loss = self._compute_loss(R)
                print(f"Iteration {iteration+1}/{self.n_iterations}, Loss: {loss:.4f}")
        
        print(f"\nTraining completed!")
        print(f"  - User embeddings shape: {self.user_factors.shape}")
        print(f"  - Item embeddings shape: {self.item_factors.shape}")
        
        return self
    
    def _solve_factors(self, R, fixed_factors, reg):
        """Solve least squares for one set of factors"""
        n = R.shape[0]
        k = fixed_factors.shape[1]
        factors = np.zeros((n, k))
        
        YtY = fixed_factors.T @ fixed_factors
        reg_eye = reg * np.eye(k)
        
        for i in range(n):
            r_i = R[i].toarray().flatten()
            idx = r_i > 0
            
            if idx.sum() == 0:
                continue
            
            Y_i = fixed_factors[idx]
            r_i_nonzero = r_i[idx]
            
            factors[i] = np.linalg.solve(
                Y_i.T @ Y_i + reg_eye,
                Y_i.T @ r_i_nonzero
            )
        
        return factors
    
    def _compute_loss(self, R):
        """Calculate reconstruction error + regularization"""
        predictions = self.user_factors @ self.item_factors.T
        R_dense = R.toarray()
        mask = R_dense > 0
        diff = (R_dense - predictions) * mask
        mse = np.sum(diff ** 2) / R.nnz
        
        reg_loss = self.reg_param * (
            np.sum(self.user_factors ** 2) + 
            np.sum(self.item_factors ** 2)
        )
        
        return mse + reg_loss
    
    def get_similar_items(self, item_id, top_k=10):
        """Get similar items based on item factors"""
        if item_id not in self.item_map:
            return []
        
        item_idx = self.item_map[item_id]
        item_vec = self.item_factors[item_idx]
        
        item_vec_norm = item_vec / (np.linalg.norm(item_vec) + 1e-8)
        all_items_norm = self.item_factors / (
            np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = all_items_norm @ item_vec_norm
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [self.reverse_item_map[idx] for idx in top_indices]
    
    def save(self, path="models/als_model.pkl"):
        model_data = {
            'item_factors': self.item_factors,      
            'item_map': self.item_map,              
            'reverse_item_map': self.reverse_item_map,  
            'n_factors': self.n_factors,
            'n_iterations': self.n_iterations,
            'reg_param': self.reg_param
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to {path}")
        import os
        file_size = os.path.getsize(path) / 1e6
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Items: {len(self.item_map)}")
    
    @classmethod
    def load(cls, path="models/als_model.pkl"):
        """Load model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            n_factors=model_data['n_factors'],
            n_iterations=model_data['n_iterations'],
            reg_param=model_data['reg_param']
        )
        model.item_factors = model_data['item_factors']
        model.item_map = model_data['item_map']
        model.reverse_item_map = model_data['reverse_item_map']
        
        print(f"✓ Model loaded from {path}")
        print(f"  - Items: {len(model.item_map)}")
        print(f"  - Factors: {model.n_factors}")
        
        return model


def evaluate_item_similarity(model, test_df, k=10):
    """
    Đánh giá item-to-item similarity
    Kiểm tra xem similar items có xuất hiện cùng trong test set không
    """
    print(f"\nEvaluating item-to-item similarity...")
    
    # Lấy các items có nhiều interactions
    item_counts = test_df['product_id'].value_counts()
    popular_items = item_counts[item_counts >= 3].index[:50]  # Top 50 items
    
    hits = 0 #số users mua trùng
    total = 0
    
    for item_id in popular_items:
        if item_id not in model.item_map:
            continue
        
        # Lấy users đã mua item này
        users_bought_this = set(test_df[test_df['product_id'] == item_id]['user_id'])
        
        # Lấy similar items
        similar_items = model.get_similar_items(item_id, top_k=k)
        
        # Kiểm tra xem có user nào mua cả 2 không
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
    print(f"  - Explanation: Similar items có chung users trong {accuracy*100:.1f}% cases")
    
    return accuracy


def train_als_model():
    
    # Load data
    print("\n1. Loading data...")
    dataset, unique_user_ids, unique_product_ids, _, _, _, _ = load_tf_data()
    
    # Convert to DataFrame
    interactions = []
    for batch in dataset.batch(1000):
        for i in range(len(batch['user_id'])):
            interactions.append({
                'user_id': batch['user_id'][i].numpy().decode('utf-8'),
                'product_id': batch['product_id'][i].numpy().decode('utf-8')
            })
    
    df = pd.DataFrame(interactions)
    print(f"Dataset loaded")
    print(f"Total interactions: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique products: {df['product_id'].nunique()}")
    print(f"Avg interactions/product: {len(df)/df['product_id'].nunique():.1f}")
    
    # Split data
    print("\n2. Splitting data...")
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)} interactions")
    print(f"Test:  {len(test_df)} interactions")
    
    # Train ALS
    print("\n3. Training ALS model...")
    
    als_model = AlternatingLeastSquares(
        n_factors=50,
        n_iterations=15,
        reg_param=0.01
    )
    
    als_model.fit(train_df)
    
    # Evaluate
    print("\n4. Evaluation")

    evaluate_item_similarity(als_model, test_df, k=10)
    
    # Save model 
    print("\n5. Saving model...")

    als_model.save("models/als_model.pkl")
    
    # Demo similar items
    print("\n6. Demo Item-to-Item Similarity")
    print("="*60)
    sample_items = df['product_id'].value_counts().head(3).index
    
    for sample_item in sample_items:
        similar = als_model.get_similar_items(sample_item, top_k=5)
        print(f"\nSimilar items to '{sample_item}':")
        for i, product_id in enumerate(similar, 1):
            print(f"  {i}. {product_id}")
    

    print("TRAINING DONE")


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_als_model()