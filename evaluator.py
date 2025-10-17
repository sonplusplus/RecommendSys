import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


class RecommenderEvaluator:
    
    def __init__(self, model, test_df):
        self.model = model
        self.test_df = test_df.copy()
        
        self.test_df['user_id'] = self.test_df['user_id'].astype(str)
        self.test_df['product_id'] = self.test_df['product_id'].astype(str)
        
        self.ground_truth = defaultdict(set)
        for _, row in test_df.iterrows():
            self.ground_truth[row['user_id']].add(row['product_id'])
    
    def hit_rate(self, k=10, sample_size=None):
        users = list(self.ground_truth.keys())
        
        if sample_size and sample_size < len(users):
            users = np.random.choice(users, sample_size, replace=False)
        
        hits = 0
        total = 0
        
        for user_id in tqdm(users, desc=f"Hit Rate@{k}", leave=False):
            if user_id not in self.model.user_map:
                continue
            
            recs = self.model.predict_for_user(user_id, top_k=k)
            if recs is None:
                continue
            
            ground_truth_items = self.ground_truth[user_id]
            if any(item in ground_truth_items for item in recs):
                hits += 1
            
            total += 1
        
        hr = hits / total if total > 0 else 0
        
        return {
            'hit_rate': hr,
            'hits': hits,
            'total': total,
            'percentage': f"{hr*100:.2f}%"
        }
    
    def precision_recall_f1(self, k=10, sample_size=None):
        users = list(self.ground_truth.keys())
        
        if sample_size and sample_size < len(users):
            users = np.random.choice(users, sample_size, replace=False)
        
        precisions = []
        recalls = []
        
        for user_id in tqdm(users, desc=f"Precision/Recall@{k}", leave=False):
            if user_id not in self.model.user_map:
                continue
            
            recs = self.model.predict_for_user(user_id, top_k=k)
            if recs is None:
                continue
            
            ground_truth_items = self.ground_truth[user_id]
            relevant_recommended = len([r for r in recs if r in ground_truth_items])
            
            prec = relevant_recommended / k if k > 0 else 0
            recall = relevant_recommended / len(ground_truth_items) if len(ground_truth_items) > 0 else 0
            
            precisions.append(prec)
            recalls.append(recall)
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': f1,
            'precision_percentage': f"{avg_precision*100:.2f}%",
            'recall_percentage': f"{avg_recall*100:.2f}%",
            'num_users': len(precisions)
        }
    
    def ndcg(self, k=10, sample_size=None):
        users = list(self.ground_truth.keys())
        
        if sample_size and sample_size < len(users):
            users = np.random.choice(users, sample_size, replace=False)
        
        ndcg_scores = []
        
        for user_id in tqdm(users, desc=f"NDCG@{k}", leave=False):
            if user_id not in self.model.user_map:
                continue
            
            recs = self.model.predict_for_user(user_id, top_k=k)
            if recs is None:
                continue
            
            ground_truth_items = self.ground_truth[user_id]
            
            dcg = 0
            for i, item in enumerate(recs, 1):
                relevance = 1 if item in ground_truth_items else 0
                dcg += relevance / np.log2(i + 1)
            
            ideal_relevances = [1] * min(len(ground_truth_items), k)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
            
            ndcg_score = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg_score)
        
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        
        return {
            'ndcg@k': avg_ndcg,
            'percentage': f"{avg_ndcg*100:.2f}%",
            'num_users': len(ndcg_scores)
        }
    
    def coverage(self, k=10, sample_size=None):
        users = list(self.ground_truth.keys())
        
        if sample_size and sample_size < len(users):
            users = np.random.choice(users, sample_size, replace=False)
        
        recommended_items = set()
        
        for user_id in tqdm(users, desc=f"Coverage@{k}", leave=False):
            if user_id not in self.model.user_map:
                continue
            
            recs = self.model.predict_for_user(user_id, top_k=k)
            if recs:
                recommended_items.update(recs)
        
        total_items = len(self.model.item_map)
        coverage_rate = len(recommended_items) / total_items if total_items > 0 else 0
        
        return {
            'coverage': coverage_rate,
            'unique_recommended': len(recommended_items),
            'total_items': total_items,
            'percentage': f"{coverage_rate*100:.2f}%"
        }
    
    def diversity(self, k=10, sample_size=50):
        users = list(self.ground_truth.keys())
        
        if sample_size and sample_size < len(users):
            users = np.random.choice(users, sample_size, replace=False)
        
        diversity_scores = []
        
        for user_id in tqdm(users, desc=f"Diversity@{k}", leave=False):
            if user_id not in self.model.user_map:
                continue
            
            recs = self.model.predict_for_user(user_id, top_k=k)
            if recs is None or len(recs) < 2:
                continue
            
            distances = []
            for i in range(len(recs)):
                for j in range(i+1, len(recs)):
                    item_i = recs[i]
                    item_j = recs[j]
                    
                    if item_i in self.model.item_map and item_j in self.model.item_map:
                        idx_i = self.model.item_map[item_i]
                        idx_j = self.model.item_map[item_j]
                        
                        vec_i = self.model.item_factors[idx_i]
                        vec_j = self.model.item_factors[idx_j]
                        
                        similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-8)
                        distance = 1 - similarity
                        distances.append(distance)
            
            if distances:
                diversity_scores.append(np.mean(distances))
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
        
        return {
            'diversity': avg_diversity,
            'percentage': f"{avg_diversity*100:.2f}%",
            'num_users': len(diversity_scores)
        }
    
    def evaluate_all(self, k=10, sample_size=None):
        print(f"Evaluating @{k} (sample_size={sample_size or 'all'})...")
        
        results = {
            'k': k,
            'sample_size': sample_size or len(self.ground_truth),
            'hit_rate': self.hit_rate(k, sample_size),
            'precision_recall': self.precision_recall_f1(k, sample_size),
            'ndcg': self.ndcg(k, sample_size),
            'coverage': self.coverage(k, sample_size),
            'diversity': self.diversity(k, min(sample_size or 50, 50))
        }
        
        return results
    
    def plot_metrics_comparison(self, k_values=[5, 10, 20, 50], sample_size=100):
        metrics = {
            'hit_rate': [],
            'precision': [],
            'recall': [],
            'ndcg': [],
            'coverage': []
        }
        
        for k in k_values:
            results = self.evaluate_all(k, sample_size)
            
            metrics['hit_rate'].append(results['hit_rate']['hit_rate'])
            metrics['precision'].append(results['precision_recall']['precision@k'])
            metrics['recall'].append(results['precision_recall']['recall@k'])
            metrics['ndcg'].append(results['ndcg']['ndcg@k'])
            metrics['coverage'].append(results['coverage']['coverage'])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Metrics vs K', fontsize=16)
        
        axes[0, 0].plot(k_values, metrics['hit_rate'], marker='o', linewidth=2)
        axes[0, 0].set_title('Hit Rate@K')
        axes[0, 0].set_xlabel('K')
        axes[0, 0].set_ylabel('Hit Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(k_values, metrics['precision'], marker='o', label='Precision', linewidth=2)
        axes[0, 1].plot(k_values, metrics['recall'], marker='s', label='Recall', linewidth=2)
        axes[0, 1].set_title('Precision & Recall@K')
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(k_values, metrics['ndcg'], marker='o', color='green', linewidth=2)
        axes[1, 0].set_title('NDCG@K')
        axes[1, 0].set_xlabel('K')
        axes[1, 0].set_ylabel('NDCG')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(k_values, metrics['coverage'], marker='o', color='red', linewidth=2)
        axes[1, 1].set_title('Coverage@K')
        axes[1, 1].set_xlabel('K')
        axes[1, 1].set_ylabel('Coverage')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics