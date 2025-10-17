import pandas as pd
import pickle
from mf import AlternatingLeastSquares
from evaluator import RecommenderEvaluator


def evaluate_trained_model():
    """Evaluate trained ALS model"""
    
    try:
        als_model = AlternatingLeastSquares.load("models/als_model.pkl")
    except FileNotFoundError:
        print("Model not found at models/als_model.pkl")
        return
    
    try:
        test_df = pd.read_csv("data/test_interactions.csv")
    except FileNotFoundError:
        df = pd.read_csv("data/interactions.csv")
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    test_df['user_id'] = test_df['user_id'].astype(str)
    test_df['product_id'] = test_df['product_id'].astype(str)
    
    print(f"Test: {len(test_df):,} interactions | {test_df['user_id'].nunique():,} users | {test_df['product_id'].nunique():,} products")
    
    evaluator = RecommenderEvaluator(als_model, test_df)
    results = evaluator.evaluate_all(k=10, sample_size=100)
    
    with open("evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    results_summary = {
        'metric': [
            'Hit Rate@10',
            'Precision@10', 
            'Recall@10',
            'F1-Score@10',
            'NDCG@10',
            'Coverage',
            'Diversity'
        ],
        'value': [
            results['hit_rate']['hit_rate'],
            results['precision_recall']['precision@k'],
            results['precision_recall']['recall@k'],
            results['precision_recall']['f1@k'],
            results['ndcg']['ndcg@k'],
            results['coverage']['coverage'],
            results['diversity']['diversity']
        ]
    }
    
    results_df = pd.DataFrame(results_summary)
    results_df['percentage'] = results_df['value'].apply(lambda x: f"{x*100:.2f}%")
    results_df.to_csv("evaluation_summary.csv", index=False)
    
    try:
        evaluator.plot_metrics_comparison(k_values=[5, 10, 20, 50], sample_size=50)
    except Exception as e:
        print(f"Plot failed: {e}")
    

    print("eval result:")
    print(results_df.to_string(index=False))
    
    hr = results['hit_rate']['hit_rate']
    prec = results['precision_recall']['precision@k']
    ndcg = results['ndcg']['ndcg@k']
    cov = results['coverage']['coverage']
    
    status = []
    if hr > 0.5:
        status.append(f"hit rate ({hr*100:.1f}%):good ")
    elif hr > 0.3:
        status.append(f"hit rate ({hr*100:.1f}%):fair")
    else:
        status.append(f"hit rate ({hr*100:.1f}%):low .try alpha=60 or n_factors(dim_vec)=100")
    
    if prec > 0.1:
        status.append(f"precision ({prec*100:.1f}%): ok")
    else:
        status.append(f"precision ({prec*100:.1f}%): low. more data")
    
    if ndcg > 0.5:
        status.append(f"NDCG ({ndcg*100:.1f}%): good")
    elif ndcg > 0.3:
        status.append(f"NDCG ({ndcg*100:.1f}%): mid")
    else:
        status.append(f"NDCG ({ndcg*100:.1f}%): low")
    
    if cov > 0.3:
        status.append(f"coverage ({cov*100:.1f}%): good")
    elif cov > 0.1:
        status.append(f"coverage ({cov*100:.1f}%): mid")
    else:
        status.append(f"coverage ({cov*100:.1f}%): low → focus on pop item")
    
    print("\n".join(status))

    
    return results


if __name__ == "__main__":
    import os
    
    if not os.path.exists("models/als_model.pkl"):
        print("train als model")
        exit(1)
    
    evaluate_trained_model()