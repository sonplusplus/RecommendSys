import tensorflow as tf
import tensorflow_recommenders as tfrs
from load_data import load_tf_data
import numpy as np
import os

dataset, unique_user_ids, unique_product_ids, unique_categories, unique_brands, product_prices, product_descriptions = load_tf_data()

print(f"Dataset: {len(unique_user_ids)} users, {len(unique_product_ids)} products\n")

tf.random.set_seed(42)
shuffled = dataset.shuffle(10_000, seed=42, reshuffle_each_iteration=False)

total_size = len(shuffled)
train_size = int(total_size * 0.72)
val_size = int(total_size * 0.08)

train = shuffled.take(train_size)
val = shuffled.skip(train_size).take(val_size)
test = shuffled.skip(train_size + val_size)

embed_d = 32

# USER TOWER - Giảm regularization
user_t = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
    tf.keras.layers.Embedding(
        len(unique_user_ids) + 1, embed_d,
        embeddings_regularizer=tf.keras.regularizers.l2(0.0001)  # Giảm từ 0.001
    )
], name="user_tower")

# ITEM TOWER - Giảm regularization và dropout
product_id_input = tf.keras.Input(shape=(), dtype=tf.string, name='product_id')
category_input = tf.keras.Input(shape=(), dtype=tf.string, name='category')
brand_input = tf.keras.Input(shape=(), dtype=tf.string, name='brand')
price_input = tf.keras.Input(shape=(), dtype=tf.float32, name='price')
description_input = tf.keras.Input(shape=(), dtype=tf.string, name='description')

product_emb = tf.keras.layers.Embedding(
    len(unique_product_ids) + 1, embed_d,
    embeddings_regularizer=tf.keras.regularizers.l2(0.0001)
)(tf.keras.layers.StringLookup(vocabulary=unique_product_ids)(product_id_input))
product_emb = tf.keras.layers.Flatten()(product_emb)

category_emb = tf.keras.layers.Embedding(
    len(unique_categories) + 1, 8,
    embeddings_regularizer=tf.keras.regularizers.l2(0.0001)
)(tf.keras.layers.StringLookup(vocabulary=unique_categories)(category_input))
category_emb = tf.keras.layers.Flatten()(category_emb)

brand_emb = tf.keras.layers.Embedding(
    len(unique_brands) + 1, 8,
    embeddings_regularizer=tf.keras.regularizers.l2(0.0001)
)(tf.keras.layers.StringLookup(vocabulary=unique_brands)(brand_input))
brand_emb = tf.keras.layers.Flatten()(brand_emb)

price_log = tf.math.log1p(price_input)
norm_price = tf.keras.layers.Normalization(axis=None)
norm_price.adapt(np.log1p(product_prices))
proc_price = norm_price(price_log)
proc_price = tf.keras.layers.Reshape((1,))(proc_price)

text_vector = tf.keras.layers.TextVectorization(
    max_tokens=500, output_sequence_length=40,
    standardize='lower_and_strip_punctuation'
)
text_vector.adapt(product_descriptions)
vectorized_desc = text_vector(description_input)
desc_emb = tf.keras.layers.Embedding(500, 8)(vectorized_desc)
desc_emb = tf.keras.layers.GlobalAveragePooling1D()(desc_emb)
desc_emb = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -10.0, 10.0))(desc_emb)

combined = tf.keras.layers.Concatenate()([
    product_emb, category_emb, brand_emb, proc_price, desc_emb
])
combined = tf.keras.layers.Dropout(0.1)(combined)  # Giảm từ 0.4 xuống 0.1
output = tf.keras.layers.Dense(embed_d)(combined)

item_t = tf.keras.Model(
    inputs={
        'product_id': product_id_input,
        'category': category_input,
        'brand': brand_input,
        'price': price_input,
        'description': description_input
    },
    outputs=output,
    name="item_tower"
)

# TWO TOWER MODEL
class TwoTowerModel(tfrs.Model):
    def __init__(self, user_model, item_model):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=dataset.map(lambda x: {
                    'product_id': x['product_id'],
                    'category': x['category'],
                    'brand': x['brand'],
                    'price': x['price'],
                    'description': x['description']
                }).batch(128).map(item_model),
                ks=[1, 5, 10, 50, 100]
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            ),
            temperature=0.1
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        item_features = {k: features[k] for k in ['product_id', 'category', 'brand', 'price', 'description']}
        item_embeddings = self.item_model(item_features, training=training)
        return self.task(user_embeddings, item_embeddings, compute_metrics=not training)

model = TwoTowerModel(user_t, item_t)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))  # Tăng từ 0.001

# Checkpoint callback
os.makedirs("models/checkpoints", exist_ok=True)

class TowerCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_total_loss')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.model.user_model.save(f"models/checkpoints/user_tower_best")
            self.model.item_model.save(f"models/checkpoints/item_tower_best")
            print(f"\nSaved best towers at epoch {epoch + 1} (val_loss: {val_loss:.4f})")

tower_checkpoint = TowerCheckpoint()

print("Training with Adam optimizer (lr=0.01) - 20 epochs\n")
history = model.fit(
    train.batch(128),
    validation_data=val.batch(128),
    epochs=20,  # Giảm từ 50 xuống 20
    callbacks=[tower_checkpoint],
    verbose=2
)

# Load best towers
print("\nLoading best towers...")
best_user_t = tf.keras.models.load_model("models/checkpoints/user_tower_best")
best_item_t = tf.keras.models.load_model("models/checkpoints/item_tower_best")

# Rebuild model với best towers
model_best = TwoTowerModel(best_user_t, best_item_t)
model_best.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

train_losses = history.history['total_loss']
val_losses = history.history['val_total_loss']

print(f"\nBest epoch: {tower_checkpoint.best_epoch}")
print(f"  Best val loss: {tower_checkpoint.best_val_loss:.4f}")

print(f"\nFinal epoch (20):")
print(f"  Train loss: {train_losses[-1]:.4f}")
print(f"  Val loss: {val_losses[-1]:.4f}")
print(f"  Gap: {val_losses[-1] - train_losses[-1]:.4f}")

# Test
print("\n" + "="*60)
print("EVALUATING ON TEST SET (using best model)")
print("="*60)

test_results = model_best.evaluate(test.batch(128), return_dict=True, verbose=0)
print(f"\nTest metrics:")
print(f"  Loss: {test_results['total_loss']:.4f}")
print(f"  Top-1 accuracy: {test_results.get('factorized_top_k/top_1_categorical_accuracy', 0):.4f}")
print(f"  Top-5 accuracy: {test_results.get('factorized_top_k/top_5_categorical_accuracy', 0):.4f}")
print(f"  Top-10 accuracy: {test_results.get('factorized_top_k/top_10_categorical_accuracy', 0):.4f}")
print(f"  Top-50 accuracy: {test_results.get('factorized_top_k/top_50_categorical_accuracy', 0):.4f}")
print(f"  Top-100 accuracy: {test_results.get('factorized_top_k/top_100_categorical_accuracy', 0):.4f}")

# Save final models
print("\n" + "="*60)
print("SAVING FINAL MODELS")
print("="*60)

model_best.item_model.save("models/item_tower")
model_best.user_model.save("models/user_tower")
print("User tower saved")
print("Item tower saved")

# Create index
index = tfrs.layers.factorized_top_k.BruteForce(model_best.user_model)
all_products = dataset.map(lambda x: {k: x[k] for k in ['product_id', 'category', 'brand', 'price', 'description']})
index.index_from_dataset(
    tf.data.Dataset.zip((
        all_products.batch(100).map(lambda x: x["product_id"]),
        all_products.batch(100).map(model_best.item_model)
    ))
)
_ = index(tf.constant([unique_user_ids[0]]))
index.save("models/rs_model")
print("Index saved")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best model from epoch {tower_checkpoint.best_epoch} saved")
print("Models location: models/user_tower, models/item_tower, models/rs_model")