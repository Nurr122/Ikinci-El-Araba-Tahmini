import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Sayfa Yapılandırması
st.set_page_config(page_title="Pro Eksper v2", layout="wide", page_icon="🚗")

# --- CSS: RENKLERİ VE GÖRÜNÜRÜLÜĞÜ KESİN OLARAK SABİTLE ---
st.markdown("""
    <style>
    /* Ana Ekran Arka Planı */
    .stApp { background-color: #f5f7f9 !important; }

    /* Rapor Kartı ve İçindeki Her Şey (Yazıları Siyaha Zorla) */
    .report-card { 
        background-color: #ffffff !important; 
        padding: 25px; 
        border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        border: 1px solid #d1d5db;
        margin-top: 25px;
        color: #000000 !important; /* Tüm metinleri siyah yapar */
    }

    /* Kartın içindeki başlık (📋 Detaylı Eksper Analizi) için özel siyah ayarı */
    .report-card h4 { 
        color: #000000 !important; 
        font-weight: bold !important;
        margin-bottom: 15px !important;
    }

    /* Kartın içindeki diğer metinler, listeler ve alt başlıklar */
    .report-card b, .report-card p, .report-card div {
        color: #000000 !important;
    }

    /* Ana ekrandaki Subheader (Teknik Özellikler vb.) için siyah ayarı */
    h1, h2, h3, h4, h5, h6, label, p, span {
        color: #1a1a1a !important;
    }

    /* Sol Panel (Sidebar) Beyaz Yazı / Siyah Arka Plan */
    [data-testid="stSidebar"] { background-color: #111111 !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }

    /* Buton Tasarımı */
    .stButton>button { 
        background-color: #ff4b4b !important; 
        color: white !important; 
        font-weight: bold;
        border-radius: 8px;
    }
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
    return model, scaler, model_cols, m, s, md, r, sh, ks, ck, km

model, scaler, model_columns, markalar, seriler, modeller, renkler, sehirler, kasalar, cekisler, kimdenler = load_assets()

st.title("🚗 Araba Değerleme Eksper Raporu")

# --- GİRİŞLER ---
with st.sidebar:
    st.header("📋 Araç Bilgileri")
    marka_s = st.selectbox("Marka", markalar)
    seri_s = st.selectbox("Seri", seriler)
    model_s = st.selectbox("Model Detayı", modeller)
    yil = st.number_input("Model Yılı", 1990, 2026, 2018)
    km = st.number_input("Kilometre", 0, 1000000, 85000)
    sehir_s = st.selectbox("Şehir", sehirler)
    renk_s = st.selectbox("Renk", renkler)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("⚙️ Teknik Özellikler")
    vites = st.radio("Vites Tipi", ["Manuel", "Otomatik", "Yarı Otomatik"], horizontal=True)
    yakit = st.radio("Yakıt", ["Benzin", "Dizel", "Elektrik", "Hibrit", "LPG & Benzin"], horizontal=True)
    kasa = st.selectbox("Kasa Tipi", kasalar)
    cekis = st.selectbox("Çekiş", cekisler)
    motor_h = st.number_input("Motor Hacmi (cc)", 600, 6000, 1598)
    motor_g = st.number_input("Motor Gücü (HP)", 50, 600, 110)

with col_b:
    st.subheader("🛠️ Kondisyon & Hasar")
    tramer = st.number_input("Tramer Kaydı (TL)", 0, 1000000, 0)
    boya = st.slider("Boyalı Parça Sayısı", 0, 13, 0)
    degisen = st.slider("Değişen Parça Sayısı", 0, 13, 0)
    kimden = st.selectbox("Satıcı Türü", kimdenler)
    tuketim = st.number_input("Ort. Yakıt Tüketimi (lt/100km)", 1.0, 25.0, 5.5)
    depo = st.number_input("Yakıt Deposu (lt)", 10, 150, 50)

# --- ANALİZ ---
if st.button("💰 ANALİZİ BAŞLAT"):
    with st.spinner('Piyasa analizi yapılıyor...'):
        time.sleep(1)
        
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        input_df['marka'], input_df['seri'], input_df['model'] = markalar.index(marka_s), seriler.index(seri_s), modeller.index(model_s)
        input_df['yil'], input_df['kilometre'], input_df['renk'], input_df['sehir'] = yil, km, renkler.index(renk_s), sehirler.index(sehir_s)
        input_df['tramer'], input_df['boyali_sayisi'], input_df['degisen_sayisi'] = tramer, boya, degisen
        input_df['motor_hacmi_temiz'], input_df['motor_gucu_temiz'] = motor_h, motor_g
        input_df['ortalama_yakit_tuketimi'], input_df['yakit_deposu'] = tuketim, depo

        if vites != "Manuel": input_df[f'vites_tipi_{vites}'] = 1
        if yakit != "Benzin": input_df[f'yakit_tipi_{yakit}'] = 1
        if f'kasa_tipi_{kasa}' in model_columns: input_df[f'kasa_tipi_{kasa}'] = 1
        if f'cekis_{cekis}' in model_columns: input_df[f'cekis_{cekis}'] = 1
        if f'kimden_{kimden}' in model_columns: input_df[f'kimden_{kimden}'] = 1

        input_scaled = scaler.transform(input_df)
        final_price = np.expm1(model.predict(input_scaled))[0]

        st.divider()
        low, mid, high = st.columns(3)
        with low: st.metric("Minimum Piyasa", f"{final_price*0.93:,.0f} TL")
        with mid: st.success(f"### Ortalama Değer\n ## {final_price:,.2f} TL")
        with high: st.metric("Maksimum Piyasa", f"{final_price*1.07:,.0f} TL")

        # --- RAPOR KARTI ---
        st.markdown(f"""
        <div class="report-card">
            <h4>📋 Detaylı Eksper Analizi</h4>
            <p>Seçilen <b>{marka_s} {seri_s} ({yil})</b> aracınızın analizi tamamlanmıştır.</p>
            <hr style="border: 0.5px solid #eee;">
            <div style='display: flex; justify-content: space-between;'>
                <div>
                    <b>🛠️ Teknik Künye:</b><br>
                    • Motor: {motor_h}cc / {motor_g}HP<br>
                    • Tüketim: {tuketim} lt<br>
                </div>
                <div>
                    <b>🚗 Kondisyon:</b><br>
                    • Kilometre: {km:,.0f} km<br>
                    • Tramer: {tramer:,.0f} TL
                </div>
            </div>
            <hr style="border: 0.5px solid #eee;">
            <p style="color: #ff4b4b !important; font-weight: bold;">
                Uzman Görüşü: Bu araç için piyasa normu {final_price:,.0f} TL seviyesidir.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
