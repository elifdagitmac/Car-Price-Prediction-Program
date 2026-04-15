import pandas as pd
import sys
import os

# Klasör yolunu garantiye alıyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Importları en temiz haliyle yapıyoruz
from src.data_prep import load_and_clean_data, encode_categorical_data, split_data
from src.model import train_model, make_predictions
import src.utils as utils

def main():
    print("🚀 Araba Segment Sınıflandırma Programı Başlıyor...")
    
    try:
        # 1. Veri Hazırlığı
        csv_path = "car_data.csv"
        df = load_and_clean_data(csv_path)
        df_encoded, mappings = encode_categorical_data(df)
        X_train, X_test, y_train, y_test = split_data(df_encoded)
        
        # 2. Model Eğitimi
        model = train_model(X_train, y_train)
        
        # 3. Başarıyı Ölç ve Görselleştir
        y_tahmin = make_predictions(model, X_test)
        
        # Fonksiyonları utils üzerinden çağırıyoruz (Hata almamak için en güvenli yol)
        utils.evaluate_model_performance(y_test, y_tahmin)
        utils.plot_confusion_matrix(y_test, y_tahmin)
        utils.plot_sigmoid_curve(model, X_test, y_test)
        # Modelin hangi özelliklere (beygir, motor vb.) ne kadar puan verdiğini gösterir
        ozellik_isimleri = X_train.columns.tolist()
        utils.plot_feature_importance(model, ozellik_isimleri)
        # 2. Korelasyon Isı Haritası
        # Verilerin birbiriyle olan ilişkisini gösterir
        utils.plot_correlation_heatmap(df_encoded)

        print("\n--- 🚗 ARAÇ SEGMENT TAHMİN SİSTEMİ ---")
        
        while True:
            devam = input("\nYeni bir tahmin başlatılsın mı? (Enter / q): ").strip().lower()
            if devam == 'q':
                break
            
            try:
                car_name = input("1) Araç Marka/Model: ").strip()
                f_type = input("2) Yakıt Tipi (gas/diesel): ").strip().lower()
                c_body = input("3) Gövde Tipi (sedan/hatchback/convertible/hardtop/wagon): ").strip().lower()
                h_power = float(input("4) Beygir Gücü (Horsepower): ").strip())
                e_size = float(input("5) Motor Hacmi (Engine Size): ").strip())

                if f_type in mappings['fueltype'] and c_body in mappings['carbody']:
                    input_row = pd.DataFrame({
                        'fueltype': [mappings['fueltype'][f_type]],
                        'carbody': [mappings['carbody'][c_body]],
                        'horsepower': [h_power],
                        'enginesize': [e_size]
                    })

                    result = model.predict(input_row)[0]
                    status = "💎 LÜKS / PAHALI" if result == 1 else "🛒 EKONOMİK"
                    
                    print(f"\n--- 📝 TAHMİN RAPORU: {car_name.upper()} ---")
                    print(f"✅ Sonuç: Bu araç {status} segmentinde değerlendirildi.")
                else:
                    print("❌ Hata: Girdiğiniz yakıt veya gövde tipi veri setinde bulunamadı!")

            except ValueError:
                print("❌ Hata: Sayısal alanlara lütfen sadece rakam girin.")
            except Exception as e:
                print(f"❌ Beklenmedik bir hata: {e}")

    except Exception as e:
        print(f"🔥 Program başlatılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()