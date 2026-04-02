import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Verideki boş satırları temizle
    df = df.dropna()
    return df

def split_data(df):
    # Senin listende 'Price' yazdığı için burayı 'Price' yapıyoruz
    y = df['Price'] 
    X = df.drop('Price', axis=1) # Fiyat dışındaki her şey özellik (X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test