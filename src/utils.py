#1. blok: kütüphaneleri içe aktardık 
import matplotlib.pyplot as plt#tahmin ve gerçek fiyatları karşılaştıran grafiği çizmek için 
import joblib#modeli kaydetmek ve yüklemek için kullanılan bir kütüphane
import os #işletim sistemi ile ilgili işlemler için kullanılan bir kütüphane 

#2. blok: grafik çizme 
def plot_regression_results(y_test, predictions, title='Gerçek vs Tahmin Edilen Fiyatlar'):
    
    #Modelin başarısını görselleştiren bir grafik çizer.
    #Doğru üzerindeki noktalar ne kadar yakınsa model o kadar başarılıdır.
    
    plt.figure(figsize=(10, 6)) #grafiğin boyutunu ayarlar inç cinsinden 
    plt.scatter(y_test, predictions, color='blue', alpha=0.5)#Gerçek fiyatları (y_test) x eksenine, tahmin edilen fiyatları (predictions) y eksenine yerleştirir. 
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)#Gerçek fiyatlarla tahmin edilen fiyatların eşit olduğu bir çizgi çizer. Bu çizgi, modelin ne kadar iyi tahmin ettiğini görsel olarak gösterir. Noktalar bu çizgiye ne kadar yakınsa model o kadar başarılıdır.
    plt.xlabel('Gerçek Fiyatlar')
    plt.ylabel('Tahmin Edilen Fiyatlar')
    plt.title(title)
    #eksenlerin grafiğin isimlerini ayarlar
    plt.show()#tabloyu ekranda gösterir

#3. blok: modeli kaydetme 
def save_trained_model(model, filename='car_price_model.pkl'):
    
    #Eğitilen modeli diske kaydeder ve böylece daha sonra tekrar eğitmeye gerek kalmadan kullanabiliriz.
    
    if not os.path.exists('models'):
        os.makedirs('models')
     #işletim sistemine 'models' klasörü var mı yokmu diye sorar 
     #eğer yoksa bu klasörü oluşturur. 

    filepath = os.path.join('models', filename)#klasör adıyla dosya adını birleştirip path oluşturur
    joblib.dump(model, filepath)#modelin o anki halini alır ve belirttiğimiz dosya yoluna fiziksel olarak kaydeder.
    print(f"✅ Model başarıyla kaydedildi: {filepath}")