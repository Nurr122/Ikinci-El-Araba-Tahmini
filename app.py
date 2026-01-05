import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Sayfa Yapılandırması
st.set_page_config(page_title="Pro Eksper v2", layout="wide")

# --- CSS: DARK MODE ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117 !important; }
    [data-testid="stSidebar"] { background-color: #0E1117 !important; border-right: 1px solid #262730; }
    h1, h2, h3, h4, h5, h6, p, label, span, div { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
    .stSelectbox div[data-baseweb="select"] div { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
    .stNumberInput input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
    .stRadio label { color: #FFFFFF !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #FFFFFF !important; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    header[data-testid="stHeader"] * { color: #FFFFFF !important; }
    header[data-testid="stHeader"] svg { fill: #FFFFFF !important; }
    .report-card { background-color: #262730 !important; padding: 20px; border-radius: 10px; border: 1px solid #41444C; color: #FFFFFF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .report-card hr { border-color: #555555 !important; }
    .stButton>button { background-color: #ff4b4b !important; color: white !important; font-weight: bold; border: none; transition: all 0.3s; }
    .stButton>button:hover { background-color: #ff2b2b !important; box-shadow: 0 0 10px rgba(255, 75, 75, 0.5); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('araba_fiyat_catboost_model.pkl')
    scaler = joblib.load('araba_scaler.pkl')
    model_cols = joblib.load('model_columns.pkl')
    
    # İndeks bulmak için düz listeler
    m = joblib.load('marka_listesi.pkl')
    s = joblib.load('seri_listesi.pkl')
    md = joblib.load('model_listesi.pkl')
    r = joblib.load('renk_listesi.pkl')
    sh = joblib.load('sehir_listesi.pkl')
    ks = joblib.load('kasa_listesi.pkl')
    ck = joblib.load('cekis_listesi.pkl')
    km = joblib.load('kimden_listesi.pkl')
    
    # YENİ 7 KATMANLI AĞAÇ
    agac = joblib.load('arac_agaci_v2.pkl')
    
    return model, scaler, model_cols, m, s, md, r, sh, ks, ck, km, agac

(cat_model, scaler, model_columns, markalar_list, seriler_list, modeller_list, 
 renkler, sehirler, kasalar_list, cekisler, kimdenler, arac_agaci) = load_assets()

st.title("Araba Değerleme Eksper Raporu")

# --- SOL PANEL: SIRALI SEÇİM ---
with st.sidebar:
    st.header("Araç Seçimi")
    
    # 1. YIL
    yil = st.selectbox("Yıl", options=arac_agaci.keys())
    
    # 2. MARKA (Seçilen Yıla Göre)
    marka_s = st.selectbox("Marka", options=arac_agaci[yil].keys())
    
    # 3. SERİ (Seçilen Yıl ve Markaya Göre)
    seri_s = st.selectbox("Seri", options=arac_agaci[yil][marka_s].keys())
    
    # 4. MODEL (Seçilen Yıl, Marka ve Seriye Göre)
    model_s = st.selectbox("Model", options=arac_agaci[yil][marka_s][seri_s].keys())
    
    # 5. KASA TİPİ
    kasa_s = st.selectbox("Kasa Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s].keys())
    
    # 6. YAKIT TİPİ
    yakit_s = st.selectbox("Yakıt Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s][kasa_s].keys())
    
    # 7. VİTES TİPİ
    vites_s = st.selectbox("Vites Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s][kasa_s][yakit_s])

# --- ANA EKRAN: DETAYLAR ---
st.subheader("Detaylar ve Kondisyon")

col1, col2, col3 = st.columns(3)

with col1:
    km = st.number_input("Kilometre", 0, 1000000, 85000)
    renk_s = st.selectbox("Renk", renkler)
    cekis = st.selectbox("Çekiş", cekisler)
    sehir_s = st.selectbox("Şehir", sehirler)

with col2:
    motor_h = st.number_input("Motor Hacmi (cc)", 600, 6000, 1598)
    motor_g = st.number_input("Motor Gücü (HP)", 50, 1000, 110)
    tuketim = st.number_input("Ort. Yakıt Tüketimi (lt)", 1.0, 30.0, 5.5)
    depo = st.number_input("Yakıt Deposu (lt)", 20, 150, 50)

with col3:
    kimden = st.selectbox("Kimden", kimdenler)
    tramer = st.number_input("Tramer Kaydı (TL)", 0, 5000000, 0)
    boya = st.slider("Boyalı Parça", 0, 13, 0)
    degisen = st.slider("Değişen Parça", 0, 13, 0)

st.markdown("---")

if st.button("FİYATI HESAPLA"):
    with st.spinner('Piyasa analizi yapılıyor...'):
        time.sleep(0.5)
        
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Seçimleri indekslere çevir
        try:
            input_df['marka'] = markalar_list.index(marka_s)
            input_df['seri'] = seriler_list.index(seri_s)
            input_df['model'] = modeller_list.index(model_s)
            input_df['yil'] = yil
            input_df['kilometre'] = km
            input_df['renk'] = renkler.index(renk_s)
            input_df['sehir'] = sehirler.index(sehir_s)
            
            # Detaylar
            input_df['tramer'] = tramer
            input_df['boyali_sayisi'] = boya
            input_df['degisen_sayisi'] = degisen
            input_df['motor_hacmi_temiz'] = motor_h
            input_df['motor_gucu_temiz'] = motor_g
            input_df['ortalama_yakit_tuketimi'] = tuketim
            input_df['yakit_deposu'] = depo

            # One-Hot Encoding alanları
            if vites_s != "Manuel": input_df[f'vites_tipi_{vites_s}'] = 1
            if yakit_s != "Benzin": input_df[f'yakit_tipi_{yakit_s}'] = 1
            
            kasa_col = f'kasa_tipi_{kasa_s}'
            if kasa_col in model_columns: input_df[kasa_col] = 1
                
            if f'cekis_{cekis}' in model_columns: input_df[f'cekis_{cekis}'] = 1
            if f'kimden_{kimden}' in model_columns: input_df[f'kimden_{kimden}'] = 1

            # Tahmin
            input_scaled = scaler.transform(input_df)
            final_price = np.expm1(cat_model.predict(input_scaled))[0]

            st.success(f"### Tahmini Piyasa Değeri: {final_price:,.0f} TL")
            
            # Detaylı Kart
            st.markdown(f"""
            <div class="report-card">
                <h4> Araç Özeti</h4>
                <p><b>{yil} {marka_s} {seri_s} {model_s}</b></p>
                <p>{kasa_s} | {yakit_s} | {vites_s}</p>
                <hr>
                <div style="display: flex; justify-content: space-between;">
                    <div>Min: <b>{final_price*0.93:,.0f} TL</b></div>
                    <div>Ort: <b>{final_price:,.0f} TL</b></div>
                    <div>Max: <b>{final_price*1.07:,.0f} TL</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Hesaplama hatası: {e}")
