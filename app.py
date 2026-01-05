import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Sayfa Yapılandırması
st.set_page_config(page_title="Pro Eksper v2", layout="wide")

# --- CSS: FULL DARK MODE (TAM KARANLIK MOD) ---
st.markdown("""
    <style>
    /* 1. TÜM ARKA PLANLARI SİYAH YAP */
    .stApp {
        background-color: #0E1117 !important; /* Koyu antrasit/siyah */
    }
    
    [data-testid="stSidebar"] {
        background-color: #0E1117 !important; /* Sidebar da aynı renk */
        border-right: 1px solid #262730; /* İnce bir ayırma çizgisi */
    }

    /* 2. TÜM YAZILARI BEYAZ YAP (ZORUNLU KIL) */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* 3. INPUT ALANLARI VE WIDGET'LAR */
    /* Selectbox, Number Input vb. içindeki yazılar */
    .stSelectbox div[data-baseweb="select"] div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    .stNumberInput input {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    /* Metrik Değerleri (Fiyatlar) */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }

    /* 4. SAĞ ÜSTTEKİ DEPLOY VE MENU */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    header[data-testid="stHeader"] * {
        color: #FFFFFF !important; /* Üst menü yazıları beyaz */
    }
    header[data-testid="stHeader"] svg {
        fill: #FFFFFF !important; /* Üç nokta ikonu beyaz */
    }

    /* 5. RAPOR KARTI (KOYU GRİ KUTU) */
    .report-card { 
        background-color: #262730 !important; /* Arka plandan biraz daha açık gri */
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #41444C;
        color: #FFFFFF !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Kartın içindeki HR çizgisi */
    .report-card hr {
        border-color: #555555 !important;
    }

    /* 6. BUTON TASARIMI */
    .stButton>button { 
        background-color: #ff4b4b !important; 
        color: white !important; 
        font-weight: bold; 
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2b2b !important;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
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

st.title("Araba Değerleme Eksper Raporu")

# --- SOL PANEL ---
with st.sidebar:
    st.header("Araç Bilgileri")
    marka_s = st.selectbox("Marka", markalar)
    seri_s = st.selectbox("Seri", seriler)
    model_s = st.selectbox("Model", modeller)
    yil = st.number_input("Yıl", 1990, 2026, 2018)
    km = st.number_input("Kilometre", 0, 1000000, 85000)
    sehir_s = st.selectbox("Şehir", sehirler)
    renk_s = st.selectbox("Renk", renkler)

# --- ANA EKRAN ---
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Teknik Özellikler")
    vites = st.radio("Vites Tipi", ["Manuel", "Otomatik", "Yarı Otomatik"], horizontal=True)
    yakit = st.radio("Yakıt", ["Benzin", "Dizel", "Elektrik", "Hibrit", "LPG & Benzin"], horizontal=True)
    kasa = st.selectbox("Kasa Tipi", kasalar)
    cekis = st.selectbox("Çekiş", cekisler)
    motor_h = st.number_input("Motor Hacmi (cc)", 600, 6000, 1598)
    motor_g = st.number_input("Motor Gücü (HP)", 50, 600, 110)

with col_b:
    st.subheader("Kondisyon & Hasar")
    tramer = st.number_input("Tramer Kaydı (TL)", 0, 1000000, 0)
    boya = st.slider("Boyalı Parça Sayısı", 0, 13, 0)
    degisen = st.slider("Değişen Parça Sayısı", 0, 13, 0)
    kimden = st.selectbox("Kimden", kimdenler)
    tuketim = st.number_input("Ort. Yakıt Tüketimi (lt/100km)", 1.0, 25.0, 5.5)
    depo = st.number_input("Yakıt Deposu (lt)", 10, 150, 50)

if st.button("Fiyat Tahmin Et"):
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

        # Rapor Kartı da koyu renk olacak
        st.markdown(f"""
        <div class="report-card">
            <h4>📋 Detaylı Eksper Analizi</h4>
            <p>Seçilen <b>{marka_s} {seri_s} ({yil})</b> aracınız için analiz tamamlandı.</p>
            <hr>
            <p style="color: #ff4b4b !important; font-weight: bold;">
                Uzman Görüşü: Aracın teknik donanımı ve kondisyonu göz önüne alındığında ortalama değer {final_price:,.0f} TL'dir.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
