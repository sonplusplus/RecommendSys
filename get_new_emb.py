"""
Nếu muốn tạo embedding cho sp mới thì chạy file này
Chạy xong sẽ in ra vector embedding của sp mới
Quan trọng: phải đảm bảo có item_tower trong thư mục rs_model
"""

import tensorflow as tf
import numpy as np

print("load item_tower ")
item_tower = tf.saved_model.load("rs_model/item_tower")
print("load success")

new_prod ={
    #fake data for example
    'product_id': np.array(['999999']),
    'category': np.array(['Electronics']),
    'brand': np.array(['BrandX']),
    'price': np.array([25000000.0], dtype=np.float32),
    'description': np.array(['Đây là 1 sản phẩm tuyệt vời'])
}

print("Creating embedding for new product...")
new_prod_emb = item_tower(new_prod)
print("Embedding created successfully:")
print(new_prod_emb.numpy())