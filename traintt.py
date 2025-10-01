import tensorflow as tf
import tensorflow_recommenders as tfrs
from load_data import load_tf_data

print("check gpu")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Đã tìm thấy và cấu hình để sử dụng GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("Cant find GPU, CPU instead")
print("-------------------------------------\n")



#load data
dataset, unique_user_ids, unique_product_ids = load_tf_data()

if dataset is None:
    print("No data available")
    exit()

#split data 0.8 train, 0.2 test
tf.random.set_seed(42)
shuffled = dataset.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train_size = int(len(shuffled) * 0.8)
train = shuffled.take(train_size)
test = shuffled.skip(train_size)

#size of vector embeddings
embed_d = 32

#user tower
user_t = tf.keras.Sequential([
    #string user_id to unique num (first layer)
    tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None
    ),
    #change number represent for user to vector (second layer)
    tf.keras.layers.Embedding(input_dim=len(unique_user_ids) + 1, output_dim=embed_d)
])

#item tower
item_t = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_product_ids, mask_token=None
    ),
 
    tf.keras.layers.Embedding(input_dim=len(unique_product_ids) + 1, output_dim=embed_d)
])

#model two tower
class TwoTowerRecommender(tfrs.Model):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=dataset.batch(128).map(self.item_tower)
            )
        )

    def compute_loss(self, data, training=False):
        user_embeddings = self.user_tower(data["user_id"])
        item_embeddings = self.item_tower(data["product_id"])
        return self.task(user_embeddings, item_embeddings)
    
#train model
products = dataset.map(lambda x: x["product_id"])

model = TwoTowerRecommender(user_t, item_t)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

print("Start training")
model.fit(train.batch(8192), epochs=5)

print("Evaluating")
model.evaluate(test.batch(4096), return_dict=True)

index = tfrs.layers.factorized_top_k.BruteForce(model.user_tower)
index.build(products.batch(128).map(model.item_tower))

#save model
tf.saved_model.save(index, "export_model/model")
print("Model saved at export_model/model")