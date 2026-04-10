import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Hedef Değişken (Fiyatın ortasından bölüyoruz)
    if 'price' in df.columns:
        threshold = df['price'].median()
        df['is_luxury'] = (df['price'] > threshold).astype(int)
    
    # KRİTİK: Modeli sadece senin sorduğun sorularla sınırlıyoruz
    # Böylece model başka hiçbir şeye (ağırlık, motor hacmi vb.) bakamaz
    needed_cols = ['fueltype', 'carbody', 'horsepower', 'is_luxury']
    
    # Sadece bu sütunları al, diğer her şeyi çöpe at
    df = df[needed_cols]
    
    return df.dropna()

def encode_categorical_data(df):
    le = LabelEncoder()
    mappings = {} # Kelime -> Sayı sözlüğümüz burada duracak
    
    # Sadece metin içeren sütunları seçiyoruz
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        # Sayıya çeviriyoruz
        df[col] = le.fit_transform(df[col].astype(str))
        # Sözlüğü kaydediyoruz (Örn: {'gas': 1, 'diesel': 0})
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
    return df, mappings # Hem güncellenmiş tabloyu hem sözlüğü döndürür

def split_data(df):
    y = df['is_luxury']
    X = df.drop('is_luxury', axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=42)