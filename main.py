import pandas as pd  
from src.data_prep import load_and_clean_data, encode_categorical_data, split_data
from src.model import train_model, make_predictions
from src.utils import evaluate_model_performance, plot_confusion_matrix, plot_sigmoid_curve #

def main():
    print("🚀 Araba Segment Sınıflandırma Programı Başlıyor...")
    
    # 1. Veriyi Yükle ve Temizle
    csv_path = "car_data.csv"
    df = load_and_clean_data(csv_path)
    
    # 2. Kategorik Verileri Şifrele (Encoding)
    df, mappings = encode_categorical_data(df)
    
    # 3. Veriyi Böl (Train-Test)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 4. Modeli Eğit
    model = train_model(X_train, y_train)
    
    # 5. Başarıyı Ölç ve Görselleştir
    predictions = make_predictions(model, X_test)
    evaluate_model_performance(y_test, predictions)
    plot_confusion_matrix(y_test, predictions)
    plot_sigmoid_curve(model, X_test, y_test)

    # 6. Kullanıcıdan Veri Alıp Tahmin Yapma
    print("\n--- 🚗 ARAÇ SEGMENT TAHMİN SİSTEMİ ---")
    
    while True:
        user_input = input("\nYeni bir tahmin başlatılsın mı? (Enter / q): ")
        if user_input.lower() == 'q':
            break
        
        try:
            car_name = input("1) Araç Marka/Model: ")
            f_type = input("2) Yakıt Tipi (gas/diesel): ").lower().strip()
            c_body = input("3) Gövde Tipi (sedan/hatchback/convertible/hardtop/wagon): ").lower().strip()
            h_power = float(input("4) Beygir Gücü (Horsepower): "))

            # Girdileri modele uygun hale getiriyoruz
            input_row = pd.DataFrame({
                'fueltype': [mappings['fueltype'][f_type]],
                'carbody': [mappings['carbody'][c_body]],
                'horsepower': [h_power]
            })

            # Tahmin
            result = model.predict(input_row)[0]
            
            status = "💎 LÜKS / PAHALI" if result == 1 else "🛒 EKONOMİK"
            print(f"\n--- 📝 TAHMİN RAPORU: {car_name.upper()} ---")
            print(f"✅ Sonuç: Bu araç {status} segmentinde değerlendirildi.")
                
        except KeyError as e:
            print(f"❌ Hata: Geçersiz seçim! Lütfen belirtilen seçeneklerden birini girin.")
        except ValueError:
            print("❌ Hata: Beygir gücü için sadece sayı girin!")
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()