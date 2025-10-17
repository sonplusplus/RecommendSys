import pandas as pd
import pickle
from mf import AlternatingLeastSquares
from evaluator import RecommenderEvaluator


def evaluate_trained_model():


    
    # 1. Load model
    print("\n1. Loading trained model...")
    try:
        als_model = AlternatingLeastSquares.load("models/als_model.pkl")
    except FileNotFoundError:
        print("Model als cant be found")
        return
    
    
    try:
        # Thử load test set riêng
        test_df = pd.read_csv("data/test_interactions.csv")
        print(f"Loaded test set: {len(test_df):,} interactions")
    except FileNotFoundError:
        df = pd.read_csv("data/interactions.csv")
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"✓ Created test set: {len(test_df):,} interactions")
    
    # Ensure string types
    test_df['user_id'] = test_df['user_id'].astype(str)
    test_df['product_id'] = test_df['product_id'].astype(str)
    
    print(f"   Users in test: {test_df['user_id'].nunique():,}")
    print(f"   Products in test: {test_df['product_id'].nunique():,}")
    
    # 3. Create evaluator
    print("\n3. Initializing evaluator...")
    evaluator = RecommenderEvaluator(als_model, test_df)
    
    # 4. Run evaluation
    print("\n4. Running evaluation (this may take a few minutes)...")
    results = evaluator.evaluate_all(k=10, sample_size=100)
    
    # 5. Save results
    print("\n5. Saving results...")
    with open("evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Export to CSV
    results_summary = {
        'metric': [
            'Hit Rate@10',
            'Precision@10', 
            'Recall@10',
            'F1-Score@10',
            'NDCG@10',
            'Coverage'
        ],
        'value': [
            results['hit_rate']['hit_rate'],
            results['precision_recall']['precision@k'],
            results['precision_recall']['recall@k'],
            results['precision_recall']['f1@k'],
            results['ndcg']['ndcg@k'],
            results['coverage']['coverage']
        ],
        'percentage': [
            results['hit_rate']['percentage'],
            results['precision_recall']['precision_percentage'],
            results['precision_recall']['recall_percentage'],
            f"{results['precision_recall']['f1@k']*100:.2f}%",
            results['ndcg']['percentage'],
            results['coverage']['percentage']
        ]
    }
    
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv("evaluation_summary.csv", index=False)
    print("✓ Results saved to evaluation_summary.csv")
    
    # 6. Plot comparison
    print("\n6. Generating comparison plots...")
    try:
        evaluator.plot_metrics_comparison(k_values=[5, 10, 20, 50], sample_size=50)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # 7. Print summary

    print("EVALUATION SUMMARY")

    print(results_df.to_string(index=False))

    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    hr = results['hit_rate']['hit_rate']
    prec = results['precision_recall']['precision@k']
    ndcg = results['ndcg']['ndcg@k']
    cov = results['coverage']['coverage']
    
    if hr > 0.5:
        print(f"Hit Rate ({hr*100:.1f}%) is GOOD - Model finds relevant items frequently")
    elif hr > 0.3:
        print(f"Hit Rate ({hr*100:.1f}%) is FAIR - Room for improvement")
    else:
        print(f"❌ Hit Rate ({hr*100:.1f}%) is LOW - Consider tuning hyperparameters")
    
    if prec > 0.1:
        print(f"Precision ({prec*100:.1f}%) is ACCEPTABLE for implicit feedback")
    else:
        print(f"Precision ({prec*100:.1f}%) is LOW - May need more training data")
    
    if ndcg > 0.5:
        print(f"NDCG ({ndcg*100:.1f}%) shows GOOD ranking quality")
    else:
        print(f"NDCG ({ndcg*100:.1f}%) - Ranking could be better")
    
    if cov > 0.3:
        print(f"Coverage ({cov*100:.1f}%) is GOOD - Diverse recommendations")
    elif cov > 0.1:
        print(f"Coverage ({cov*100:.1f}%) is FAIR - Some diversity")
    else:
        print(f"Coverage ({cov*100:.1f}%) is LOW - Too focused on popular items")
    

    print("✓ Evaluation complete!")

    
    return results


if __name__ == "__main__":
    import os
    

    if not os.path.exists("models/als_model.pkl"):
        print("Model als not found.")
        exit(1)
    
    # Run evaluation
    results = evaluate_trained_model()