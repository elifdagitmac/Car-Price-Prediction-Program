import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Target Variable (We divide from the middle of the price)
    if 'price' in df.columns:
        threshold = df['price'].median()
        df['is_luxury'] = (df['price'] > threshold).astype(int)
    
    #  We only limit the model to the questions the user asks.
    # So the model can't look at anything else (like weight, engine size, etc.)
    needed_cols = ['fueltype', 'carbody', 'horsepower', 'is_luxury', 'enginesize']
    
    # Manual Scaling: We divide the engine displacement by 100 to put it on the same scale as horsepower
    if 'enginesize' in df.columns:
        df['enginesize'] = df['enginesize'] / 100
    # Just take these columns, throw everything else away
    df = df[needed_cols]
    
    return df.dropna()

def encode_categorical_data(df):
    le = LabelEncoder()
    mappings = {} #Our word -> number dictionary will stay here
    
    # We only select columns that contain text
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        # We're converting it to a number
        df[col] = le.fit_transform(df[col].astype(str))
        # We are saving the dictionary (e.g., {'gas': 1, 'diesel': 0})
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
    return df, mappings # It returns both the updated table and the dictionary

def split_data(df):
    y = df['is_luxury']
    X = df.drop('is_luxury', axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=42)
