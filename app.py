import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Sayfa Yapılandırması
st.set_page_config(page_title="Pro Eksper v2", layout="wide")

# --- CSS: EKSTRA GÖRÜNÜRLÜK AYARLARI ---
st.markdown("""
    <style>
    /* Rapor Kartı Tasarımı */
    .report-card {
        background-color: #262730 !important;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41444C;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    
    /* Buton Tasarımı */
    .stButton>button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-weight: bold;
        border: none;
        height: 3em;
        width: 100%;
        margin-top: 20px;
    }

    /* Başlıkların altındaki boşluğu azalt */
    .stHeader { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('araba_fiyat_catboost_model.pkl')
    scaler = joblib.load('araba_scaler.pkl')
    model_cols = joblib.load('model_columns.pkl')
    m = joblib.load('marka_listesi.pkl')
    s = joblib.load('seri_listesi.pkl')
    md = joblib.load('model_listesi.pkl')
    r = joblib.load('renk_listesi.pkl')
    sh = joblib.load('sehir_listesi.pkl')
    ks = joblib.load('kasa_listesi.pkl')
    ck = joblib.load('cekis_listesi.pkl')
    km = joblib.load('kimden_listesi.pkl')
    agac = joblib.load('arac_agaci_v2.pkl')
    return model, scaler, model_cols, m, s, md, r, sh, ks, ck, km, agac

(cat_model, scaler, model_columns, markalar_list, seriler_list, modeller_list, 
 renkler, sehirler, kasalar_list, cekisler, kimdenler, arac_agaci) = load_assets()

# --- ANA EKRAN BAŞLIK ---
st.title("Araba Değerleme Eksper Raporu")
st.write("Aracınızın bilgilerini girerek piyasa değerini anında öğrenin.")

# --- BÖLÜM 1: SIRALI ARAÇ SEÇİMİ (ANA EKRAN) ---
st.markdown("### 1. Araç Seçimi")
col_1, col_2, col_3, col_4 = st.columns(4)

with col_1:
    yil = st.selectbox("Yıl", options=arac_agaci.keys())
with col_2:
    marka_s = st.selectbox("Marka", options=arac_agaci[yil].keys())
with col_3:
    seri_s = st.selectbox("Seri", options=arac_agaci[yil][marka_s].keys())
with col_4:
    model_s = st.selectbox("Model", options=arac_agaci[yil][marka_s][seri_s].keys())

col_5, col_6, col_7 = st.columns(3)
with col_5:
    kasa_s = st.selectbox("Kasa Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s].keys())
with col_6:
    yakit_s = st.selectbox("Yakıt Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s][kasa_s].keys())
with col_7:
    vites_s = st.selectbox("Vites Tipi", options=arac_agaci[yil][marka_s][seri_s][model_s][kasa_s][yakit_s])

st.divider()

# --- BÖLÜM 2: DETAYLAR ---
st.markdown("### 2. Detaylar ve Kondisyon")
c1, c2, c3 = st.columns(3)

with c1:
    km = st.number_input("Kilometre", 0, 1000000, 85000)
    renk_s = st.selectbox("Renk", renkler)
    cekis = st.selectbox("Çekiş", cekisler)
    sehir_s = st.selectbox("Şehir", sehirler)

with c2:
    motor_h = st.number_input("Motor Hacmi (cc)", 600, 6000, 1598)
    motor_g = st.number_input("Motor Gücü (HP)", 50, 1000, 110)
    tuketim = st.number_input("Ort. Yakıt Tüketimi (lt)", 1.0, 30.0, 5.5)
    depo = st.number_input("Yakıt Deposu (lt)", 20, 150, 50)

with c3:
    kimden = st.selectbox("Kimden", kimdenler)
    tramer = st.number_input("Tramer Kaydı (TL)", 0, 5000000, 0)
    boya = st.slider("Boyalı Parça", 0, 13, 0)
    degisen = st.slider("Değişen Parça", 0, 13, 0)

# --- ANALİZ BUTONU ---
if st.button("FİYATI HESAPLA"):
    with st.spinner('Piyasa analizi yapılıyor...'):
        time.sleep(0.5)
        try:
            input_df = pd.DataFrame(0, index=[0], columns=model_columns)
            input_df['marka'] = markalar_list.index(marka_s)
            input_df['seri'] = seriler_list.index(seri_s)
            input_df['model'] = modeller_list.index(model_s)
            input_df['yil'], input_df['kilometre'] = yil, km
            input_df['renk'], input_df['sehir'] = renkler.index(renk_s), sehirler.index(sehir_s)
            input_df['tramer'], input_df['boyali_sayisi'], input_df['degisen_sayisi'] = tramer, boya, degisen
            input_df['motor_hacmi_temiz'], input_df['motor_gucu_temiz'] = motor_h, motor_g
            input_df['ortalama_yakit_tuketimi'], input_df['yakit_deposu'] = tuketim, depo

            # Akıllı One-Hot Encoding
            for col, val in [('vites_tipi_', vites_s), ('yakit_tipi_', yakit_s), ('kasa_tipi_', kasa_s), ('cekis_', cekis), ('kimden_', kimden)]:
                full_col = f"{col}{val}"
                if full_col in model_columns: input_df[full_col] = 1

            input_scaled = scaler.transform(input_df)
            final_price = np.expm1(cat_model.predict(input_scaled))[0]

            st.success(f"### Tahmini Piyasa Değeri: {final_price:,.0f} TL")
            
            st.markdown(f"""
            <div class="report-card">
                <h4>📋 Araç Özeti</h4>
                <p><b>{yil} {marka_s} {seri_s} {model_s}</b></p>
                <p>{kasa_s} | {yakit_s} | {vites_s}</p>
                <hr>
                <div style="display: flex; justify-content: space-between; color: white;">
                    <div>Min: <b>{final_price*0.93:,.0f} TL</b></div>
                    <div>Ort: <b>{final_price:,.0f} TL</b></div>
                    <div>Max: <b>{final_price*1.07:,.0f} TL</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
