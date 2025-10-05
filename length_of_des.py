#
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("products.csv")
df['desc_length'] = df['description'].str.split().str.len()
print(df['desc_length'].describe())

#display diagram
plt.hist(df['desc_length'], bins=50)
plt.title('Phân phối độ dài mô tả sản phẩm')
plt.xlabel('Số từ')
plt.ylabel('Số lượng sản phẩm')
plt.show()