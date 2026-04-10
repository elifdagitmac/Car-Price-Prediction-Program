from sklearn.linear_model import LogisticRegression
import pandas as pd

# 1. Blok: Modeli Oluşturma ve Eğitme
def train_model(X_train, y_train):
    # Lojistik Regresyon modeli (Sınıflandırma yapar)
    # max_iter=1000 koyuyoruz ki model veriyi öğrenmek için yeterli turu atsın
    model = LogisticRegression(max_iter=1000)
    
    # Eğitme işlemi: Özellikler ile 'Lüks mü?' hedefi arasındaki sınırı öğrenir
    model.fit(X_train, y_train)
    return model

# 2. Blok: Tahmin Yapma
def make_predictions(model, X_test):
    # Artık çıktı fiyat değil; 0 (Ekonomik) veya 1 (Lüks) dönecek
    predictions = model.predict(X_test)
    return predictions