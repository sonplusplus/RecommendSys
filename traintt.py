from pyexpat import features
import tensorflow as tf
import tensorflow_recommenders as tfrs
from load_data import load_tf_data
import numpy as np

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
dataset, unique_user_ids, unique_product_ids, unique_categories, unique_brands, product_prices, product_descriptions = load_tf_data()
if dataset is None:
    print("No data available")
    exit()

#split data 0.8 train, 0.2 test
tf.random.set_seed(42)
shuffled = dataset.shuffle(10_000, seed=42, reshuffle_each_iteration=False)

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
product_id_input = tf.keras.Input(shape = (1,), dtype = tf.string, name = 'product_id')
category_input = tf.keras.Input(shape = (1,), dtype = tf.string, name = 'category')
brand_input = tf.keras.Input(shape = (1,), dtype = tf.string, name = 'brand')
price_input = tf.keras.Input(shape = (1,), dtype = tf.float32, name = 'price')
description_input = tf.keras.Input(shape = (1,), dtype = tf.string, name = 'description')

#embedding layers
product_id_emb = tf.keras.layers.Embedding(
    len(unique_product_ids) + 1, embed_d)(tf.keras.layers.StringLookup(vocabulary=unique_product_ids)(product_id_input)
)
category_emb = tf.keras.layers.Embedding(
    len(unique_categories) + 1, embed_d)(tf.keras.layers.StringLookup(vocabulary=unique_categories)(category_input)
)
brand_emb = tf.keras.layers.Embedding(
    len(unique_brands) + 1, embed_d)(tf.keras.layers.StringLookup(vocabulary=unique_brands)(brand_input)
)

#normalize price
norm_price = tf.keras.layers.Normalization(axis=None)
norm_price.adapt(np.array(product_prices))
proc_price = norm_price(price_input)

#Description text embedding

max_tokens = 1000   #max words in vocab

#75% is 80.5 words, so if dataset change, need to rerun length_of_des.py to get new one
out_seq_len = 81    #constant num of words in 1 desc
text_vector = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens, 
    output_sequence_length=out_seq_len
)
text_vector.adapt(product_descriptions)
vectorized_desc = text_vector(description_input)
desc_emb = tf.keras.layers.Embedding(max_tokens,embed_d,mask_zero=True)(vectorized_desc)    
desc_emb = tf.keras.layers.GlobalAveragePooling1D()(desc_emb)

#merge all features
all_f = tf.keras.layers.Concatenate()([
    tf.keras.layers.Flatten()(product_id_emb),
    tf.keras.layers.Flatten()(category_emb),
    tf.keras.layers.Flatten()(brand_emb),
    proc_price,
    desc_emb
])

#Dense layers
dense_out = tf.keras.layers.Dense(256, activation='relu')(all_f)
final_out = tf.keras.layers.Dense(embed_d)(dense_out)

#model for item
item_t = tf.keras.Model(
    inputs={
        'product_id': product_id_input,
        'category': category_input,
        'brand': brand_input,
        'price': price_input,
        'description': description_input
    },
    outputs=final_out
)

#model two tower
class TwoTowerRecommender(tfrs.Model):
    def __init__(self, user_t, item_t):
        super().__init__()
        self.user_tower = user_t
        self.item_tower = item_t


        all_prod_f = dataset.map(lambda x: {
            'product_id': x['product_id'],    
            'category': x['category'],
            'brand': x['brand'],
            'price': x['price'],
            'description': x['description']
        })
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=all_prod_f.batch(128).map(self.item_tower)
                )
        )
    def compute_loss(self, data, training=False):
        user_emb = self.user_tower(data['user_id'])
        
        #make a dict to feed item tower
        item_emb = {
            'product_id': data['product_id'],
            'category': data['category'],
            'brand': data['brand'],
            'price': data['price'],
            'description': data['description']
        }
        return self.task(user_emb,item_emb,compute_metrics=not training)


model = TwoTowerRecommender(user_t, item_t)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

print("Start training")
model.fit(train.batch(8192), epochs=10)

print("Evaluating")
model.evaluate(test.batch(4096), return_dict=True)

#save model
print("\n Save model")
index = tfrs.layers.factorized_top_k.BruteForce(model.user_tower)

all_prod_f_for_index = dataset.map(lambda x: {
            'product_id': x['product_id'],    
            'category': x['category'],
            'brand': x['brand'],
            'price': x['price'],
            'description': x['description']
        })

#index tìm kiếm, truy xuất
index.index_from_dataset(
    tf.data.Dataset.zip((
        all_prod_f_for_index.batch(100),
        all_prod_f_for_index.batch(100).map(model.item_tower)
    ))
)

#save index
tf.saved_model.save(index, "rs_model")
print("Model saved at rs_model")