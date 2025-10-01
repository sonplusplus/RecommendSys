import pandas as pd
from sqlachemy import create_engine, exc
import tensorflow as tf
import json

#Config
DB_USER = 'sonchoida'
DB_PASSWORD = 'sonchoida'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'ecommerce'

DB_URL = f'mysql+mysqlclient://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(DB_URL)

def load_tf_data():
    """Load and preprocess data from DB
    if fail, read Json instead
    """
    interactions_df = None
    try:
        query = """SELECT
        CAST(o.user_id AS CHAR) AS user_id,
        CAST(oi.product_id AS CHAR) AS product_id,
        FROM Orders o
        JOIN OrderItems oi ON o.id = oi.order_id
        """
        df = pd.read_sql(query, engine)
        print("succes load data from sql")
    except exc.SQLAlchemyError as e:
        print(f"Error loading data from DB: {e}, Loading from JSON file")
        df = None
    except Exception as e:
        print(f"Unexpected error: {e}, Loading from JSON file")
        df = None
    
    if df is None:
        try:
            with open('interactions.json', 'r',encoding='utf-8') as f:
                data = json.load(f)
            interactions_df = pd.DataFrame(data)
            interactions_df['user_id'] = interactions_df['user_id'].astype(str)
            interactions_df['product_id'] = interactions_df['product_id'].astype(str)
            
            #load products data in file json if cant connect to sql 
            with open('products.json', 'r',encoding='utf-8') as f:
                products_data = json.load(f)
            products_df = pd.DataFrame(products_data)
            products_df['id'] = products_df['id'].astype(str)

            df = pd.merge(interactions_df, products_df, on = 'product_id')
        except FileNotFoundError:
            print("Cant find file")
            return None, None, None
        except Exception as e:
            print(f"Error loading data from JSON: {e}")
            return None, None, None
        
    # Preprocess data
    #list unique ids for embedding layers
    
    unique_user_ids = df.user_id.unique()
    unique_product_ids = df.product_id.unique()
    unique_categories = df.category.unique()

    dataset_dict = {name : tf.convert_to_tensor(value) for name, value in df.items()}
    dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
    
    return dataset,unique_user_ids, unique_product_ids, unique_categories,df['price'].values