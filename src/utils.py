import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression

def evaluate_model_performance(y_test, predictions):
    # Doğruluk oranı: Yüzde kaç arabayı doğru sınıfa koyduk?
    score = accuracy_score(y_test, predictions)
    print(f"\n--- LOJİSTİK REGRESYON BAŞARI METRİĞİ ---")
    print(f"Doğruluk Oranı (Accuracy): %{score * 100:.2f}")
    return score

def plot_confusion_matrix(y_test, predictions):
    # Hangi sınıfta hata yaptığımızı gösteren tablo
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu') # Göz alıcı bir renk
    plt.title('Hata Matrisi (0=Ekonomik, 1=Lüks)')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.savefig('hata_matrisi.png')
    plt.show(block=True)

    
def plot_sigmoid_curve(model, X_test, y_test):
    """Lojistik Regresyon'un S-Eğrisi (Sigmoid) mantığını görselleştirir."""
    
    # 1. Modelin test verileri için hesapladığı olasılıkları (probabilities) alalım
    # predict_proba, [0 olasılığı, 1 olasılığı] şeklinde bir matris döndürür.
    # Biz '1' (Lüks) olma olasılığını alıyoruz.
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # 2. Gerçek sınıfları (0 veya 1) olasılıklara göre sıralayalım
    # Bu, S-eğrisinin düzgün görünmesini sağlar.
    sorted_indices = np.argsort(probabilities)
    sorted_probabilities = probabilities[sorted_indices]
    sorted_y_test = y_test.iloc[sorted_indices].values
    
    # 3. Grafiği Çizelim
    plt.figure(figsize=(10, 6))
    
    # S-Eğrisini (Sigmoid) çiziyoruz (Tahmin Edilen Olasılıklar)
    plt.plot(sorted_probabilities, sorted_probabilities, color='blue', linewidth=3, label='Sigmoid Eğrisi (Olasılık Skoru)')
    
    # Gerçek Veri Noktalarını (0 veya 1) çiziyoruz
    # Olasılık 0.5'ten büyükse Lüks (1), küçükse Ekonomik (0)
    plt.scatter(sorted_probabilities, sorted_y_test, color='red', alpha=0.5, label='Gerçek Veri Noktaları (0/1)')
    
    # 0.5 Eşik Değerini Çizelim (Karar Sınırı)
    plt.axhline(y=0.5, color='green', linestyle='--', label='Karar Eşiği (0.5)')
    
    plt.title('Lojistik Regresyon S-Eğrisi (Sigmoid) ve Karar Mantığı')
    plt.xlabel('Modelin Hesapladığı Lüks Olma Olasılığı')
    plt.ylabel('Gerçek Sınıf (0=Ekonomik, 1=Lüks)')
    plt.legend()
    plt.grid(True)
    plt.savefig('sigmoid_egrisi.png')
    print("✅ S-Eğrisi grafiği 'sigmoid_egrisi.png' olarak kaydedildi.")
    plt.show(block=True)

    
    
import pandas as pd

def plot_feature_importance(model, feature_names):
    """Lojistik Regresyon katsayılarını (coefficients) bar chart olarak görselleştirir."""
    # Katsayıları alıp mutlak değerlerini hesaplıyoruz (önem derecesi için)
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.coef_[0] # Katsayılar (pozitif lüksü, negatif ekonomiyi destekler)
    })
    
    # Mutlak değere göre sıralayalım
    importances['Abs_Importance'] = importances['Importance'].abs()
    importances = importances.sort_values(by='Abs_Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    plt.title('Lojistik Regresyon: Değişken Önem Sırası (Coefficients)')
    plt.xlabel('Katsayı Değeri (Lüksü Destekleme Derecesi)')
    plt.ylabel('Teknik Özellik')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # Sıfır çizgisi
    plt.grid(axis='x', alpha=0.3)
    
    # Kaydet ve Göster
    plt.savefig('feature_importance.png')
    print("✅ Değişken Önem grafiği 'feature_importance.png' olarak kaydedildi.")
    plt.show(block=True)

def plot_correlation_heatmap(df):
    """Veri setindeki değişkenler arası ilişkiyi (correlation) gösteren ısı haritası."""
    # Sadece sayısal sütunları seçelim
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Teknik Özellikler Arası İlişki Matrisi (Correlation Heatmap)')
    
    # Kaydet ve Göster
    plt.savefig('correlation_heatmap.png')
    print("✅ Korelasyon Isı Haritası 'correlation_heatmap.png' olarak kaydedildi.")
    plt.show(block=True)