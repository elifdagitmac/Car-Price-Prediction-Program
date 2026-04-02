from src.data_prep import load_and_clean_data, split_data

try:
    df = load_and_clean_data('car_data.csv')
    # Hangi sütun isimleri var, terminalde görelim:
    print("🔍 Dosyadaki Sütun İsimleri:", df.columns.tolist()) 
    
    # Burası hata verebilir, şimdilik sütun isimlerini görmek için çalıştırıyoruz
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("✅ Başarılı!")
except Exception as e:
    print(f"❌ Bir hata oluştu: {e}")