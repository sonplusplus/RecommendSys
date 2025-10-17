import pandas as pd

interactions = pd.read_csv('data/interactions.csv')
interactions['user_id'] = interactions['user_id'].astype(str)
interactions['product_id'] = interactions['product_id'].astype(str)

def check_cooccurrence(item_a, item_b):
    users_a = set(interactions[interactions['product_id'] == item_a]['user_id'])
    users_b = set(interactions[interactions['product_id'] == item_b]['user_id'])
    
    common = users_a & users_b
    

    print(f"Item A: {item_a}")
    print(f"  - Total users bought: {len(users_a)}")
    
    print(f"\nItem B: {item_b}")
    print(f"  - Total users bought: {len(users_b)}")
    
    print(f"\nCo-occurrence:")
    print(f"  - Users bought BOTH: {len(common)}")
    
    if len(users_a) > 0 and len(users_b) > 0:
        overlap_a = len(common) / len(users_a) * 100
        overlap_b = len(common) / len(users_b) * 100
        print(f"  - % of A users also bought B: {overlap_a:.1f}%")
        print(f"  - % of B users also bought A: {overlap_b:.1f}%")
    
    if len(common) > 0:
        print(f"\nCommon users: {sorted(list(common))[:10]}")
    else:
        print(f"\n⚠️  NO OVERLAP - These items never bought together!")

# Test các similar items từ ALS
print("Testing ALS similar items for Apple Watch (109346)")
similar_items = ['68859', '109698', '105476', '72337', '88564']

for item in similar_items:
    check_cooccurrence('109346', item)