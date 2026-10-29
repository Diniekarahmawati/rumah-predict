import json
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="RumahWajar — Dashboard Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. CUSTOM CSS (Bento Grid, Dark/Light Native Styling & Micro-Interactions)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Styling Elegan Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 28px 32px;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    
    /* Bento Card Container */
    .bento-card {
        background-color: var(--background-secondary, #ffffff);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .bento-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }

    /* Badges Status Kewajaran Harga */
    .badge-wajar {
        background-color: #d1fae5;
        color: #065f46;
        padding: 8px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-overpriced {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 8px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-underpriced {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 8px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Format Teks Metric */
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e1b4b;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. LOAD MODEL MACHINE LEARNING (.joblib)
# ---------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    model_path = Path(__file__).resolve().parent / "model_terbaik.joblib"
    return joblib.load(model_path)

model = load_ml_pipeline()
info_path = Path(__file__).resolve().parent / "info_model.json"
with info_path.open(encoding="utf-8") as file:
    model_info = json.load(file)

# ---------------------------------------------------------
# 4. SIDEBAR CONTROL (Multi-Language & Toggle Setup)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/duotone/96/4f46e5/real-estate.png", width=64)
    st.title("RumahWajar UI")
    
    language = st.radio("🌐 Bahasa / Language", ["Bahasa Indonesia", "English"], horizontal=True)
    st.markdown("---")
    
    st.markdown("### 👥 Kelompok 3 (TEK B)")
    st.caption("• Dini Eka Rahmawati (Analis Data)")
    st.caption("• Siti Lativa Irmiyati (Data Engineer & Modeler)")
    st.caption("• Lukiman Hakim (Evaluator & Developer)")
    st.markdown("---")
    st.caption("🎓 Sekolah Vokasi IPB — 2026")

# Kamus Bahasa Singkat
is_id = (language == "Bahasa Indonesia")

# ---------------------------------------------------------
# 5. HERO BANNER HEADER
# ---------------------------------------------------------
hero_title = "🏠 RumahWajar Dashboard" if is_id else "🏠 RumahWajar Dashboard"
hero_desc = (
    "Sistem Estimasi, Cek Kewajaran, dan Rekomendasi Area Harga Rumah Jabodetabek Berbasis Machine Learning"
    if is_id else
    "Machine Learning-Based System for House Price Estimation, Fairness Checking, and Area Recommendation in Jabodetabek"
)

st.markdown(f"""
<div class="hero-banner">
    <h1 style="color: white; margin: 0; font-size: 2.1rem; font-weight: 800;">{hero_title}</h1>
    <p style="margin-top: 8px; opacity: 0.9; font-size: 1rem;">{hero_desc}</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. BENTO GRID LAYOUT UTAMA (2 Kolom)
# ---------------------------------------------------------
col_input, col_result = st.columns([1, 1.25], gap="large")

# === KOLOM KIRI: FORM INPUT INTERAKTIF ===
with col_input:
    st.subheader("📋 " + ("Kalkulator Spesifikasi Rumah" if is_id else "Property Specification Input"))
    
    tab_fisik, tab_lokasi = st.tabs([
        "📐 " + ("Fisik & Ukuran" if is_id else "Physical Details"), 
        "📍 " + ("Lokasi & Legalitas" if is_id else "Location & Legality")
    ])
    
    with tab_fisik:
        luas_tanah = st.number_input("Luas Tanah (m²)", min_value=10, max_value=2000, value=90, step=5)
        luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=10, max_value=2000, value=70, step=5)
        
        c1, c2 = st.columns(2)
        with c1:
            kamar_tidur = st.slider("Kamar Tidur", 1, 10, 3)
        with c2:
            kamar_mandi = st.slider("Kamar Mandi", 1, 10, 2)
        c3, c4, c5 = st.columns(3)
        with c3:
            jumlah_lantai = st.number_input("Jumlah Lantai", min_value=1, max_value=5, value=2)
        with c4:
            carport = st.number_input("Carport", min_value=0, max_value=10, value=1)
        with c5:
            garasi = st.number_input("Garasi", min_value=0, max_value=10, value=0)
        daya_listrik = st.selectbox(
            "Daya Listrik (VA)",
            [450, 900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 10600, 11000, 13200, 16500, 23000],
            index=3,
        )

    with tab_lokasi:
        kota = st.selectbox("Kota / Kabupaten", [
            "Jakarta Selatan", "Jakarta Timur", "Jakarta Barat", "Jakarta Utara", "Jakarta Pusat",
            "Depok", "Kota Bogor", "Kab. Bogor", "Kota Tangerang", "Tangerang Selatan", "Bekasi"
        ])
        sertifikat = st.selectbox("Sertifikat Properti", ["SHM", "HGB", "Lainnya"])
        kondisi = st.selectbox(
            "Kondisi Properti",
            ["baru", "bagus sekali", "bagus", "sudah renovasi", "butuh renovasi"],
            index=2,
        )
        perabot = st.selectbox(
            "Kondisi Perabot",
            ["unfurnished", "semi furnished", "furnished"],
            index=0,
        )
        
        # Mapping Koordinat Peta
        koordinat_kota = {
            "Jakarta Selatan": [-6.2615, 106.8106],
            "Jakarta Timur": [-6.2250, 106.9004],
            "Jakarta Pusat": [-6.1805, 106.8284],
            "Jakarta Barat": [-6.1683, 106.7588],
            "Jakarta Utara": [-6.1384, 106.8640],
            "Depok": [-6.4025, 106.7942],
            "Kota Bogor": [-6.5971, 106.8060],
            "Kab. Bogor": [-6.4789, 106.8228],
            "Kota Tangerang": [-6.1783, 106.6319],
            "Tangerang Selatan": [-6.2886, 106.7179],
            "Bekasi": [-6.2383, 106.9756]
        }
        lat_long = koordinat_kota.get(kota, [-6.2088, 106.8456])

    st.markdown("---")
    harga_penawaran = st.number_input(
        "💵 " + ("Harga Penawaran Penjual (Rp)" if is_id else "Offered Price (IDR)"), 
        min_value=100000000, 
        max_value=100000000000, 
        value=1200000000, 
        step=50000000, 
        format="%d"
    )
    
    btn_hitung = st.button("🚀 " + ("Hitung Estimasi & Cek Kewajaran" if is_id else "Calculate Estimate & Check"), use_container_width=True, type="primary")

# === KOLOM KANAN: BENTO GRID RESULT & ANALYTICS ===
with col_result:
    st.subheader("📊 " + ("Hasil Analisis & Prediksi Model" if is_id else "Model Analytics & Prediction"))
    latitude, longitude = lat_long
    input_df = pd.DataFrame([{
        "land_size_m2": luas_tanah,
        "building_size_m2": luas_bangunan,
        "bedrooms": kamar_tidur,
        "bathrooms": kamar_mandi,
        "carports": carport,
        "garages": garasi,
        "floors": jumlah_lantai,
        "electricity": daya_listrik,
        "lat": latitude,
        "long": longitude,
        "city": kota,
        "certificate": sertifikat,
        "property_condition": kondisi,
        "furnishing": perabot,
    }])[model_info["fitur"]]
    estimasi_harga = float(np.exp(model.predict(input_df)[0]))
    mape = model_info["mape_test_persen"]
    batas_bawah = estimasi_harga * (1 - mape/100)
    batas_atas = estimasi_harga * (1 + mape/100)

    # BENTO CARD 1: Metric Estimasi Utama
    st.markdown(f"""
    <div class="bento-card">
        <div class="metric-title">{"ESTIMASI HARGA PASAR (MODEL ML)" if is_id else "ESTIMATED MARKET PRICE (ML)"}</div>
        <div class="metric-value">Rp {estimasi_harga:,.0f}</div>
        <p style="color: #10b981; font-size: 0.85rem; font-weight: 600; margin: 0;">
            ✓ Akurasi Model (MAPE): ±{mape}% | Rentang Wajar: Rp {batas_bawah:,.0f} – Rp {batas_atas:,.0f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # BENTO CARD 2: Cek Kewajaran Badge Status
    st.markdown("##### " + ("Status Kewajaran Harga Penawaran" if is_id else "Offered Price Status"))
    if batas_bawah <= harga_penawaran <= batas_atas:
        st.markdown('<div class="badge-wajar">✅ HARGA WAJAR — Penawaran Sesuai Pasar</div>' if is_id else '<div class="badge-wajar">✅ FAIR PRICE</div>', unsafe_allow_html=True)
        st.caption("Harga penawaran penjual berada dalam rentang toleransi variasi pasar Jabodetabek.")
    elif harga_penawaran > batas_atas:
        st.markdown('<div class="badge-overpriced">⚠️ DI ATAS PASAR (OVERPRICED)</div>' if is_id else '<div class="badge-overpriced">⚠️ OVERPRICED</div>', unsafe_allow_html=True)
        st.caption("Harga penawaran lebih tinggi dari batas estimasi wajar model. Disarankan bernegosiasi.")
    else:
        st.markdown('<div class="badge-underpriced">💡 DI BAWAH PASAR (UNDERPRICED)</div>' if is_id else '<div class="badge-underpriced">💡 UNDERPRICED</div>', unsafe_allow_html=True)
        st.caption("Harga penawaran tergolong murah dibandingkan estimasi rata-rata pasar!")

    st.markdown("---")

    # BENTO CARD 3: Interactive Leaflet Map
    st.markdown("##### 🗺️ " + ("Visualisasi Lokasi Properti" if is_id else "Property Location Map"))
    m = folium.Map(location=lat_long, zoom_start=12, tiles="OpenStreetMap")
    folium.Marker(
        lat_long,
        popup=f"{kota} - Rp {estimasi_harga:,.0f}",
        icon=folium.Icon(color="indigo", icon="home")
    ).add_to(m)
    st_folium(m, height=240, width=700)

# ---------------------------------------------------------
# 7. METRIK TAMBAHAN & GRAFIK TREN (Bento Bottom Section)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 " + ("Tren & Statistik Pasar Properti Jabodetabek" if is_id else "Market Trends & Insights"))

g1, g2 = st.columns([1, 1])

with g1:
    st.markdown("##### Feature Importance (Pengaruh Fitur terhadap Harga)")
    feature_data = pd.DataFrame({
        'Fitur': ['Luas Tanah', 'Luas Bangunan', 'Posisi Latitude', 'Posisi Longitude', 'Daya Listrik'],
        'Tingkat Pengaruh': [0.396, 0.218, 0.071, 0.054, 0.042]
    }).sort_values(by='Tingkat Pengaruh', ascending=True)
    st.bar_chart(feature_data.set_index('Fitur'))

with g2:
    st.markdown("##### Median Harga Rumah Per Kota (Miliar Rp)")
    harga_kota_df = pd.DataFrame({
        'Kota': ['Jakarta Utara', 'Jakarta Selatan', 'Jakarta Pusat', 'Tangerang Selatan', 'Depok'],
        'Median Harga (Miliar)': [10.0, 4.5, 3.8, 1.8, 0.96]
    })
    st.dataframe(harga_kota_df, use_container_width=True, hide_index=True)