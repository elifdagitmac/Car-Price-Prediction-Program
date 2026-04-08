from sklearn.linear_model import LinearRegression
import pandas as pd

# 1. Blok: Modeli Oluşturma ve Eğitme
def train_model(X_train, y_train):
    

    # Model objesini tanımlıyoruz (Denklemi kuracak beyin)
    model = LinearRegression()#OLS algoritmasıyla çalışan bir modelin altyapısını oluşturuyoruz
    
    # Eğitme işlemi: Model, X (özellikler) ile y (fiyat) arasındaki 
    # katsayıları (beta değerlerini) burada hesaplar.
    model.fit(X_train, y_train) #bu komut sayesinde model özellikler ile fiyat arasındaki ilişkiyi öğrenir. modelin eğitim süreci başlar
    #OLS algortitması çalışmay başlar. Katsayılar burada hesaplanır ve model, bu katsayıları kullanarak gelecekteki tahminler yapabilir hale gelir
    
    return model

# 2. Blok: Tahmin Yapma

# 1. Fonksiyon: Sadece katsayıları gösterir
def get_model_coefficients(model, feature_names):
    coef_df = pd.DataFrame({
        'Özellik': feature_names,
        'Katsayi (Ağirlik)': model.coef_
    })
    print(f"\nModelin Sabit Terimi (Beta 0): {model.intercept_:.2f}")
    return coef_df

# 2. Fonksiyon: Sadece tahmin yapar
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions
   