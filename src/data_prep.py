import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



# 1. Blok: Veriyi yükleme ve temizleme
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
   # bu sütunlar modelin öğrenmesi için gerekli değil, bu yüzden onları kaldırıyoruz.
    cols_to_drop = ['Car ID', 'Brand', 'Model', 'Condition'] 
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])# silmek istediğimiz sütunları geçici bir değişkenin içerisinde koyduk ve silmeden önce gerçekten bu sütunlar verisetinde varmı diye kontrol yaptık eğer yoksa program çökebilir çünkü
    
    #car ID sütunu sadece tabaloda kaçıncı satırda olduğumuzu bildirir. çoklu regresyonun matematiksel formülüne göre fiyatı etkileyen bir faktör değildir.
    #condition sütununu da matematiksel olarak bir etkisi olmadığı için çıkardık çünkü bizim modelimiz sadece sayısal verilerle çalışacak.
    #brand ve model sütunları da araçların marka ve model bilgilerini içerir, ancak bu bilgiler sayısal verilere dönüştürülmediği sürece modelin öğrenmesi için doğrudan bir katkı sağlamaz. Bu nedenle, bu sütunları da kaldırarak modelin sadece sayısal verilere odaklanmasını sağlıyoruz.
    
    # Yaş Hesaplama 
    if 'Year' in df.columns:
        df['Vehicle_Age'] = 2026 - df['Year']
        df = df.drop(columns=['Year'])
    
    df = df.dropna() #tabloda içinde eksik veri olan satırları siler çünkü eksik veriler modelin doğruluğunu etkileyebilir 
    return df

# 2. Blok: Encoding 
def encode_categorical_data(df):
    le = LabelEncoder() # # LabelEncoder kalıbını kullanarak 'le' adında spesifik bir dönüştürücü araç tanımladık.
    #  sütun isimleri:
    categorical_cols = ['Fuel Type', 'Transmission'] 
    
    for col in categorical_cols: #categorical_cols ismi için "col" adında bir değişken tanımladık.
        if col in df.columns: #eğer "col" değişkeni df'nin sütunları arasında varsa, bir alt satıra geçilir
            df[col] = le.fit_transform(df[col]) # fit_transform() metodu, 'col' sütunundaki kategorik verileri sayısal değerlere dönüştürür ve bu yeni değerleri aynı sütuna atar.
            
    return df # güncellenmiş DataFrame'i döndürür. Bu DataFrame, kategorik sütunların sayısal değerlere dönüştürülmüş haliyle geri döner


# 3. Blok: Veriyi X ve y olarak bölme
def split_data(df):
    # DİKKAT: CSV dosmanda sütun ismi 'Price' ise 'Price' kalsın, 
    # 'Selling_Price' ise onu yaz.
    y = df['Selling_Price'] 
    X = df.drop('Selling_Price', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #biz test size belirledikten sonra train size otomatik olarak kalan kısmı alır.
    return X_train, X_test, y_train, y_test