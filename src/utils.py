import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression

def evaluate_model_performance(y_test, predictions):
    # Accuracy rate: What percentage of cars did we put in the correct class?
    score = accuracy_score(y_test, predictions)
    print(f"\n--- LOJİSTİK REGRESYON BAŞARI METRİĞİ ---")
    print(f"Doğruluk Oranı (Accuracy): %{score * 100:.2f}")
    return score

def plot_confusion_matrix(y_test, predictions):
    # Table showing which class we made a mistake in
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu') 
    plt.title('Hata Matrisi (0=Ekonomik, 1=Lüks)')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.savefig('hata_matrisi.png')
    plt.show(block=True)

    
def plot_sigmoid_curve(model, X_test, y_test):
    """Lojistik Regresyon'un S-Eğrisi (Sigmoid) mantığını görselleştirir."""
    
    # 1. take the probabilities that the model calculated for the test data
    # predict_proba returns a matrix in the form [probability of 0, probability of 1].
    # We are taking the chance of being '1' (Luxury).
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # 2. rank the actual classes (0 or 1) based on probabilities
    # This makes the S-curve look smooth.
    sorted_indices = np.argsort(probabilities)
    sorted_probabilities = probabilities[sorted_indices]
    sorted_y_test = y_test.iloc[sorted_indices].values
    
    # 3. Graph
    plt.figure(figsize=(10, 6))
    
    # We're plotting the S-curve (Sigmoid) (Predicted Probabilities)
    plt.plot(sorted_probabilities, sorted_probabilities, color='blue', linewidth=3, label='Sigmoid Eğrisi (Olasılık Skoru)')
    
    # We are plotting the actual data points (0 or 1)
    # If the probability is greater than 0.5, Luxury (1), if less, Economic (0)
    plt.scatter(sorted_probabilities, sorted_y_test, color='red', alpha=0.5, label='Gerçek Veri Noktaları (0/1)')
    
    
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
    # We take the coefficients and calculate their absolute values (for importance level)
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.coef_[0] # Coefficients (support positive luxury, negative economy)
    })
    
    # sort by absolute value
    importances['Abs_Importance'] = importances['Importance'].abs()
    importances = importances.sort_values(by='Abs_Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    plt.title('Lojistik Regresyon: Değişken Önem Sırası (Coefficients)')
    plt.xlabel('Katsayı Değeri (Lüksü Destekleme Derecesi)')
    plt.ylabel('Teknik Özellik')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # Zero line
    plt.grid(axis='x', alpha=0.3)
    
    # Save and Show
    plt.savefig('feature_importance.png')
    print("✅ Değişken Önem grafiği 'feature_importance.png' olarak kaydedildi.")
    plt.show(block=True)

def plot_correlation_heatmap(df):
    """Veri setindeki değişkenler arası ilişkiyi (correlation) gösteren ısı haritası."""
    # select the numeric columns
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Teknik Özellikler Arası İlişki Matrisi (Correlation Heatmap)')
    
    # Save and Show
    plt.savefig('correlation_heatmap.png')
    print("✅ Korelasyon Isı Haritası 'correlation_heatmap.png' olarak kaydedildi.")
    plt.show(block=True)
