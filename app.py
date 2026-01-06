import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Sayfa Yapılandırması
st.set_page_config(page_title="Pro Eksper v4", layout="wide")

# --- CSS: ÖZEL TASARIM VE KARTLAR ---
# Config.toml ile uyumlu, ekranda kaybolmayan net tasarım
st.markdown("""
    <style>
    /* Rapor Kartı Tasarımı */
    .report-card {
        background-color: #262730 !important;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #41444C;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        margin-top: 25px;
    }
    .report-card h4 { color: #FFFFFF !important; margin-bottom: 15px; }
    .report-card p { color: #E0E0E0 !important; font-size: 1.1em; }
    .report-card hr { border-color: #555555 !important; }
    
    /* Hesaplama Butonu */
    .stButton>button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-weight: bold;
        border: none;
        height: 3.5em;
        width: 100%;
        margin-top: 25px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2b2b !important;
        transform: scale(1.01);
    }
    
    /* Başlık ve alt metin */
    .main-title { text-align: center; color: #FFFFFF; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Modeli ve dönüştürücüleri yüklüyoruz
    model = joblib.load('araba_fiyat_catboost_model.pkl')
    scaler = joblib.load('araba_scaler.pkl')
    model_cols = joblib.load('model_columns.pkl')
    
    # İndeksleme için orijinal listeler (Modelin sayısal karşılıklarını bulmak için)
    m = joblib.load('marka_listesi.pkl')
    s = joblib.load('seri_listesi.pkl')
    md = joblib.load('model_listesi.pkl')
    r = joblib.load('renk_listesi.pkl') 
    sh = joblib.load('sehir_listesi.pkl')
    km_list = joblib.load('kimden_listesi.pkl')
    
    # YENİ: v4 Renk detaylı ve 8 katmanlı ağaç yapısı
    agac = joblib.load('arac_agaci_v4.pkl')
    
    return model, scaler, model_cols, m, s, md, r, sh, km_list, agac

(cat_model, scaler, model_columns, markalar_list, seriler_list, modeller_list, 
 renkler_full_list, sehirler, kimdenler, arac_agaci) = load_assets()

# --- BAŞLIK ---
st.markdown("<h1 class='main-title'>Araba Değerleme Eksper Raporu</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Aracınızın detaylarını seçin, piyasa değerini yapay zeka ile hesaplayalım.</p>", unsafe_allow_html=True)

st.divider()

# --- BÖLÜM 1: ARAÇ SEÇİMİ (SIRALI) ---
st.markdown("### 1. Araç Bilgileri")
c1, c2, c3, c4 = st.columns(4)
yil = c1.selectbox("Model Yılı", options=arac_agaci.keys())
marka_s = c2.selectbox("Marka", options=arac_agaci[yil].keys())
seri_s = c3.selectbox("Seri", options=arac_agaci[yil][marka_s].keys())
model_s = c4.selectbox("Model", options=arac_agaci[yil][marka_s][seri_s].keys())

c5, c6, c7, c8 = st.columns(4)
kasa_s = c5.selectbox("Kasa Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s].keys())
yakit_s = c6.selectbox("Yakıt Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s][kasa_s].keys())
vites_s = c7.selectbox("Vites Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s][kasa_s][yakit_s].keys())
cekis_s = c8.selectbox("Çekiş", options=arac_agaci[yil][marka_s][seri_s][model_s][kasa_s][yakit_s][vites_s].keys())

# ARKA PLANDA KULLANILACAK VERİLER (Görünmez veriler ve dinamik renk listesi)
oto_veriler = arac_agaci[yil][marka_s][seri_s][model_s][kasa_s][yakit_s][vites_s][cekis_s]

st.divider()

# --- BÖLÜM 2: KONDİSYON VE HASAR ---
st.markdown("### 2. Kondisyon & Hasar Durumu")
d1, d2, d3 = st.columns(3)

with d1:
    km = st.number_input("Kilometre", 0, 1000000, 85000)
    # DİNAMİK RENK: Sadece o araca ait renkler gelir
    renk_s = st.selectbox("Renk", options=oto_veriler['renkler']) 
    sehir_s = st.selectbox("Şehir", sehirler)

with d2:
    tramer = st.number_input("Tramer Kaydı (TL)", 0, 5000000, 0)
    boya = st.slider("Boyalı Parça Sayısı", 0, 13, 0)
    degisen = st.slider("Değişen Parça Sayısı", 0, 13, 0)

with d3:
    kimden_s = st.selectbox("Satıcı Türü", kimdenler)

st.markdown("---")

# --- HESAPLAMA MANTIĞI ---
if st.button(" FİYATI HESAPLA"):
    with st.spinner('Piyasa analizi yapılıyor...'):
        time.sleep(0.7)
        try:
            # Modele gönderilecek DataFrame hazırlığı
            input_df = pd.DataFrame(0, index=[0], columns=model_columns)
            
            # Label Encoding / Orijinal listelerdeki indeksleri bulma
            input_df['marka'] = markalar_list.index(marka_s)
            input_df['seri'] = seriler_list.index(seri_s)
            input_df['model'] = modeller_list.index(model_s)
            input_df['yil'], input_df['kilometre'] = yil, km
            input_df['renk'] = renkler_full_list.index(renk_s) # Modele orijinal indeksi gönderiyoruz
            input_df['sehir'] = sehirler.index(sehir_s)
            input_df['tramer'], input_df['boyali_sayisi'], input_df['degisen_sayisi'] = tramer, boya, degisen
            
            # GÖRÜNMEZ VERİLER: Ağaçtan gelen teknik fabrika verileri
            input_df['motor_hacmi_temiz'] = int(oto_veriler['motor_hacmi'])
            input_df['motor_gucu_temiz'] = int(oto_veriler['motor_gucu'])
            input_df['ortalama_yakit_tuketimi'] = float(oto_veriler['yakit_tuketimi'])
            input_df['yakit_deposu'] = int(oto_veriler['yakit_deposu'])

            # Akıllı One-Hot Encoding (Düz vites gibi baz kategorilerde hata vermez)
            for col, val in [('vites_tipi_', vites_s), ('yakit_tipi_', yakit_s), 
                             ('kasa_tipi_', kasa_s), ('cekis_', cekis_s), ('kimden_', kimden_s)]:
                full_col = f"{col}{val}"
                if full_col in model_columns:
                    input_df[full_col] = 1

            # Tahmin ve Ölçeklendirme
            input_scaled = scaler.transform(input_df)
            raw_prediction = np.expm1(cat_model.predict(input_scaled))[0]
            
            # TRAMER DÜZELTMESİ: İstatistiksel hatayı manuel ceza (penalty) ile çözüyoruz
            final_price = raw_prediction - (tramer * 1.0)
            if final_price < (raw_prediction * 0.5): # Fiyatın aşırı düşmesini engelleme
                final_price = raw_prediction * 0.5

# --- SONUÇ EKRANI ---
            st.success(f"### Tahmini Piyasa Değeri: {final_price:,.0f} TL")
            
            st.markdown(f"""
            <div class="report-card">
                <h4> Eksper Raporu Özeti</h4>
                <p><b>Araç:</b> {yil} {marka_s} {seri_s} {model_s}</p>
                <p><b>Özellikler:</b> {kasa_s} | {yakit_s} | {vites_s} | {cekis_s}</p>
                <hr>
                <div style="display: flex; justify-content: space-between; color: white;">
                    <div>Hızlı Satış: <br><b>{final_price*0.93:,.0f} TL</b></div>
                    <div>Ortalama Piyasa: <br><b>{final_price:,.0f} TL</b></div>
                    <div>Üst Limit: <br><b>{final_price*1.07:,.0f} TL</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"Tahmin sırasında bir hata oluştu: {e}")
