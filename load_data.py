import pandas as pd
from sqlalchemy import create_engine, exc
import tensorflow as tf

#config
DB_USER = 'sonchoida'
DB_PASSWORD = 'sonchoida'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'ecommerce'

DB_URL = f"mysql+mysqldb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

# FLAG: True = dùng DB, False = dùng CSV
USE_DB = False


def load_tf_data():
    """Load and preprocess data
    - Nếu USE_DB=True → load từ DB
    - Nếu USE_DB=False → load từ CSV
    """

    df = None

    # DB
    if USE_DB:
        try:
            query = """SELECT
            CAST(o.user_id AS CHAR) AS user_id,
            CAST(oi.product_id AS CHAR) AS product_id,
            oi.price,
            p.category
            FROM Orders o
            JOIN OrderItems oi ON o.id = oi.order_id
            JOIN Products p ON oi.product_id = p.id
            """
            df = pd.read_sql(query, engine)
            print("load data from sql")
        except exc.SQLAlchemyError as e:
            print(f"error load sql: {e}")
            return None, None, None, None, None, None, None
        except Exception as e:
            print(f"error: {e}")
            return None, None, None, None, None, None, None

    # CSV
    else:
        try:
            interactions_df = pd.read_csv("data/interactions.csv")
            products_df = pd.read_csv("data/products.csv")

            for col in ['brand', 'description', 'category']:
                if col not in products_df.columns:
                    products_df[col] = ''
                products_df[col] = products_df[col].fillna('').astype(str)

            # ép kiểu string cho chắc chắn
            interactions_df['user_id'] = interactions_df['user_id'].astype(str)
            interactions_df['product_id'] = interactions_df['product_id'].astype(str)
            products_df['product_id'] = products_df['product_id'].astype(str)

            df = pd.merge(interactions_df, products_df, on="product_id")
            print("Load data from CSV successfully")
        except FileNotFoundError:
            print(" CSV file not found")
            return None, None, None, None, None, None, None
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return None, None, None, None, None, None, None

    # ----------------------
    # Preprocess dữ liệu
    # ----------------------
    unique_user_ids = df.user_id.unique()
    unique_product_ids = df.product_id.unique()
    unique_categories = df.category.unique()
    unique_brands = df.brand.unique()

    dataset_dict = {name: tf.convert_to_tensor(value) for name, value in df.items()}
    dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)

    return dataset, unique_user_ids, unique_product_ids, unique_categories, unique_brands, df['price'].values, df['description'].values


# ========= TEST =========
if __name__ == "__main__":
    dataset, users, products, categories, brands, prices, descriptions = load_tf_data()
    if dataset is not None:
        print(f"Số user: {len(users)}")
        print(f"Số sản phẩm: {len(products)}")
        print(f"Số loại category: {len(categories)}")
        print(f"Số loại brand: {len(brands)}")
        print(f"Sample price: {prices[:5]}")
        print(f"Sample description: {descriptions[0]}")
