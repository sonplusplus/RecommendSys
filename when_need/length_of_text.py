#use this when new data appear, run it to decide which max_l in phoBERT
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/products.csv")

df['name'] = df.get('name', df.get('product_name', '')).fillna('')
df['category'] = df['category'].fillna('')
df['brand'] = df['brand'].fillna('')
df['description'] = df['description'].fillna('')

df['combined_features'] = (
    df['name'] + '. ' +
    df['name'] + '. ' +  
    df['category'] + '. ' +
    df['brand'] + '. ' +
    df['description']
)

df['text_length'] = df['combined_features'].str.split().str.len()

# Statistics
print(df['text_length'].describe())

# Display diagram
plt.hist(df['text_length'], bins=50)
plt.title('Phân phối độ dài combined features')
plt.xlabel('Số từ')
plt.ylabel('Số lượng sản phẩm')
plt.show()