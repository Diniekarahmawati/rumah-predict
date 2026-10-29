# =====================================================================
# RumahWajar — Dashboard Prediksi & Cek Kewajaran Harga Rumah Jabodetabek
# Proyek Akhir Pembelajaran Mesin · tampilan Bento Grid (light/dark, ID/EN)
#
# Semua angka di dashboard berasal dari model & data asli project:
#   - estimasi harga         : model ML (pipeline preprocessing + model)
#   - akurasi & faktor harga : dihitung ulang pada 631 rumah data uji
#   - rumah mirip, grafik    : data listing asli (data_listing.csv)
#   - cari area              : hasil notebook bagian lokasi (harga_per_area.csv, dll.)
# =====================================================================

import json
import os
from datetime import datetime

import altair as alt             # grafik (sudah ikut terpasang bersama Streamlit)
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# PENGATURAN MODEL  ->  model utama yang dipakai versi aplikasi ini
# =====================================================================
FILE_MODEL_UTAMA = "model_linear_regression.joblib"   # file model utama
FILE_INFO = "info_model_linear_regression.json"   # file info/metrik model utama
LABEL_VERSI = "Linear Regression"                 # nama versi (tampil di navbar)

FILE_SEMUA_MODEL = {
    "Linear Regression": "model_linear_regression.joblib",
    "Random Forest": "model_random_forest.joblib",
    "Gradient Boosting": "model_terbaik.joblib",
    "XGBoost": "model_xgboost.joblib",
}
FILE_SEMUA_INFO = {
    "Linear Regression": "info_model_linear_regression.json",
    "Random Forest": "info_model_random_forest.json",
    "Gradient Boosting": "info_model.json",
    "XGBoost": "info_model_xgboost.json",
}

# =====================================================================
# PENGATURAN TAMPILAN  ->  warna untuk mode terang & gelap
# =====================================================================
NAMA_BRAND = "RumahWajar"
FONT = "Inter"
NAMA_TIM = "Dini Eka Rahmawati · Siti Lativa Irmiyati · Lukiman Hakim"
TEMA = {
    "terang": {
        "latar": "#F2F0EB", "kartu": "#FFFFFF", "kartu2": "#F7F6F2", "teks": "#121212", "redup": "#6E6C67",
        "garis": "#E7E4DD", "aksen": "#121212", "aksen_teks": "#FFFFFF", "sorot": "#D7F26B", "sorot_teks": "#1B2A00",
        "hero": "#121212", "hero_teks": "#FFFFFF", "hero_redup": "#A3A3A3", "input": "#F7F6F2",
        "hijau_bg": "#E3F6E8", "hijau": "#16794A", "merah_bg": "#FDE6E1", "merah": "#B3361B",
        "kuning_bg": "#FFF3CF", "kuning": "#8A6100", "bayang": "0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.04)",
        "grafik_biasa": "#D9D6CE",
    },
    "gelap": {
        "latar": "#0B0C10", "kartu": "#15171D", "kartu2": "#1C1F27", "teks": "#F4F4F5", "redup": "#9A9FAB",
        "garis": "#262A33", "aksen": "#C6F432", "aksen_teks": "#0B0C10", "sorot": "#C6F432", "sorot_teks": "#0B0C10",
        "hero": "#1C1F27", "hero_teks": "#FFFFFF", "hero_redup": "#9A9FAB", "input": "#1C1F27",
        "hijau_bg": "rgba(52,211,153,.14)", "hijau": "#4ADE80", "merah_bg": "rgba(248,113,113,.15)", "merah": "#F87171",
        "kuning_bg": "rgba(250,204,21,.14)", "kuning": "#FACC15", "bayang": "0 1px 2px rgba(0,0,0,.3)",
        "grafik_biasa": "#3A3F4B",
    },
}

# =====================================================================
# TEKS DUA BAHASA (ID / EN)
# =====================================================================
TEKS = {
    "ID": {
        "tab_dash": "Dashboard", "tab_area": "Cari Area", "tab_model": "Model", "tab_tentang": "Tentang",
        "mode_gelap": "Mode gelap", "sapaan": "Estimasi harga rumah Jabodetabek",
        "sub_sapaan": "Ubah isian di kiri, estimasi langsung diperbarui oleh model Machine Learning.",
        "kalkulator": "Kalkulator Prediksi", "kota": "Kota/Kabupaten", "area": "Kecamatan/Area",
        "lt": "Luas tanah (m²)", "lb": "Luas bangunan (m²)", "kt": "Kamar tidur", "km": "Kamar mandi",
        "lantai": "Lantai", "listrik": "Listrik (VA)", "carport": "Carport", "garasi": "Garasi",
        "sertifikat": "Sertifikat", "kondisi": "Kondisi", "perabot": "Perabot",
        "penawaran": "Harga dari penjual (juta Rp, opsional)", "simpan": "Simpan ke riwayat",
        "tersimpan": "Tersimpan ke riwayat ✓",
        "estimasi": "Estimasi harga pasar", "kisaran": "Kisaran wajar", "per_m2": "Harga per m² bangunan",
        "akurasi": "Akurasi model", "akurasi_sub": "rumah uji meleset ≤ 20%", "mape_sub": "rata-rata kesalahan (MAPE)",
        "cek": "Cek kewajaran", "isi_penawaran": "Isi harga dari penjual untuk mengecek apakah harganya wajar.",
        "WAJAR": "WAJAR", "MAHAL": "DI ATAS PASAR", "MURAH": "DI BAWAH PASAR",
        "saran_wajar": "Harga masih dalam kisaran estimasi pasar.",
        "saran_mahal": "Ada ruang negosiasi, kecuali rumah punya keunggulan yang tidak tercatat di data.",
        "saran_murah": "Bisa jadi kesempatan bagus; periksa legalitas & kondisi fisik.",
        "selisih": "selisih", "toleransi": "toleransi", "pita": "Pita hijau = kisaran wajar",
        "faktor": "Faktor pengaruh harga", "faktor_sub": "Seberapa besar akurasi turun jika faktor diacak (data uji)",
        "f_luas": "Luas tanah & bangunan", "f_lokasi": "Lokasi", "f_ruang": "Kamar & lantai",
        "f_fasilitas": "Listrik & parkir", "f_legal": "Sertifikat, kondisi, perabot",
        "grafik": "Median harga per kota", "grafik_sub": "Dari harga iklan asli · kota terpilih disorot",
        "statistik": "Ringkasan data", "total": "rumah dianalisis", "median_m2": "median harga/m² di",
        "jumlah_area": "area di", "mirip": "Rumah mirip di data", "mirip_sub": "Iklan asli dengan spesifikasi paling mirip di",
        "harga_iklan": "harga iklan", "peta": "Peta lokasi", "peta_sub": "Lokasi iklan di",
        "riwayat": "Riwayat estimasi", "riwayat_kosong": "Belum ada riwayat. Tekan \"Simpan ke riwayat\" di kalkulator.",
        "waktu": "Waktu", "lokasi": "Lokasi", "ukuran": "LT/LB", "status": "Status", "hapus": "Hapus riwayat",
        "tanpa_penawaran": "Tanpa penawaran",
        # cari area
        "area_judul": "Cari area strategis dekat kampus & kawasan industri",
        "area_sub": "Harga memakai rumah standar 70/90 (LB 70 m², LT 90 m², 2 KT, 1 KM) agar antar-area adil dibandingkan.",
        "dekat": "Saya mencari rumah dekat", "kampus": "Kampus", "industri": "Kawasan industri", "tujuan": "Tujuan",
        "budget": "Budget (juta Rp)", "jarak": "Jarak maksimal (km)", "prioritas": "Prioritas",
        "p_dekat": "Lebih dekat", "p_seimbang": "Seimbang", "p_hemat": "Lebih hemat",
        "sekitar": "Rumah 70/90 di sekitar", "indeks": "Indeks harga", "indeks_sub": "100 = rata-rata Jabodetabek",
        "iklan_sekitar": "Median harga iklan sekitar", "dari_iklan": "iklan", "rentang": "Rentang antar-area",
        "rekomendasi": "area rekomendasi", "skor": "Skor strategis", "sisa": "sisa",
        "tidak_cocok": "Belum ada area yang cocok dengan budget dan jarak ini.", "termurah": "Rumah 70/90 termurah dalam radius",
        "data_area_kosong": "Data area belum tersedia. Upload harga_per_area.csv, titik_acuan.csv, dan ringkasan_titik_acuan.csv.",
        "Premium": "Premium", "Di atas rata-rata": "Di atas rata-rata", "Rata-rata": "Rata-rata", "Terjangkau": "Terjangkau",
        # model
        "model_judul": "Perbandingan model", "model_sub": "Semua model diuji pada 631 rumah data uji yang sama. MAPE makin kecil makin akurat.",
        "dipakai": "DIPAKAI", "estimasi_semua": "Estimasi semua model untuk rumah di kalkulator", "dari": "dari",
        "model_utama": "Model utama versi ini",
        # tentang
        "tentang_judul": "Bagaimana aplikasi ini bekerja",
        "alur": [("Data", "3.553 iklan rumah Jabodetabek (Kaggle, rumah123.com)."),
                 ("Pembersihan", "Hapus duplikat & nilai tak wajar → 3.153 rumah."),
                 ("Preprocessing", "Imputasi, log-transform, scaling, one-hot (14 → 33 kolom)."),
                 ("Modeling", "Dilatih pada log(harga), Grid Search + 5-Fold CV."),
                 ("Evaluasi", "Diuji pada 631 rumah yang belum pernah dilihat model.")],
        "keterbatasan": "Keterbatasan",
        "batas": ["Harga di data adalah harga iklan, bukan harga transaksi.",
                  "Lokasi memakai titik tengah kecamatan/area, bukan alamat persis.",
                  "Data Jakarta Pusat dan rumah > Rp10 miliar sedikit, estimasinya kurang akurat.",
                  "Estimasi ini bukan penilaian (appraisal) resmi."],
        "aturan": "Aturan cek kewajaran",
        "aturan_isi": "Selisih harga penjual terhadap estimasi dibandingkan dengan MAPE model. Di dalam toleransi = WAJAR, di atasnya = DI ATAS PASAR, di bawahnya = DI BAWAH PASAR.",
        "peringatan_besar": "Ukuran rumah ini jauh di atas sebagian besar data latih, estimasi kurang akurat.",
        "peringatan_lb": "Luas bangunan lebih besar dari luas tanah × jumlah lantai. Periksa kembali isian.",
    },
    "EN": {
        "tab_dash": "Dashboard", "tab_area": "Find Area", "tab_model": "Models", "tab_tentang": "About",
        "mode_gelap": "Dark mode", "sapaan": "Jabodetabek house price estimator",
        "sub_sapaan": "Change the inputs on the left; the Machine Learning model updates the estimate instantly.",
        "kalkulator": "Price Calculator", "kota": "City/Regency", "area": "District/Area",
        "lt": "Land size (m²)", "lb": "Building size (m²)", "kt": "Bedrooms", "km": "Bathrooms",
        "lantai": "Floors", "listrik": "Electricity (VA)", "carport": "Carport", "garasi": "Garage",
        "sertifikat": "Certificate", "kondisi": "Condition", "perabot": "Furnishing",
        "penawaran": "Seller's price (million Rp, optional)", "simpan": "Save to history",
        "tersimpan": "Saved to history ✓",
        "estimasi": "Estimated market price", "kisaran": "Fair range", "per_m2": "Price per m² of building",
        "akurasi": "Model accuracy", "akurasi_sub": "of test houses within 20% error", "mape_sub": "mean error (MAPE)",
        "cek": "Fair price check", "isi_penawaran": "Enter the seller's price to check whether it is fair.",
        "WAJAR": "FAIR", "MAHAL": "ABOVE MARKET", "MURAH": "BELOW MARKET",
        "saran_wajar": "The price is within the estimated market range.",
        "saran_mahal": "There is room to negotiate, unless the house has advantages not captured in the data.",
        "saran_murah": "Could be a good deal; check the legal documents and physical condition.",
        "selisih": "difference", "toleransi": "tolerance", "pita": "Green band = fair range",
        "faktor": "Price drivers", "faktor_sub": "How much accuracy drops when each factor is shuffled (test data)",
        "f_luas": "Land & building size", "f_lokasi": "Location", "f_ruang": "Rooms & floors",
        "f_fasilitas": "Electricity & parking", "f_legal": "Certificate, condition, furnishing",
        "grafik": "Median price by city", "grafik_sub": "From real listing prices · selected city highlighted",
        "statistik": "Data summary", "total": "houses analysed", "median_m2": "median price/m² in",
        "jumlah_area": "areas in", "mirip": "Similar houses in the data", "mirip_sub": "Real listings most similar to your input in",
        "harga_iklan": "listing price", "peta": "Location map", "peta_sub": "Listing locations in",
        "riwayat": "Estimate history", "riwayat_kosong": "No history yet. Press \"Save to history\" in the calculator.",
        "waktu": "Time", "lokasi": "Location", "ukuran": "Land/Bldg", "status": "Status", "hapus": "Clear history",
        "tanpa_penawaran": "No offer",
        "area_judul": "Find strategic areas near campuses & industrial estates",
        "area_sub": "Prices use a standard 70/90 house (70 m² building, 90 m² land, 2 bed, 1 bath) for a fair comparison.",
        "dekat": "I'm looking for a house near", "kampus": "Campus", "industri": "Industrial estate", "tujuan": "Destination",
        "budget": "Budget (million Rp)", "jarak": "Max distance (km)", "prioritas": "Priority",
        "p_dekat": "Closer", "p_seimbang": "Balanced", "p_hemat": "Cheaper",
        "sekitar": "70/90 house around", "indeks": "Price index", "indeks_sub": "100 = Jabodetabek average",
        "iklan_sekitar": "Median listing price nearby", "dari_iklan": "listings", "rentang": "Range across areas",
        "rekomendasi": "recommended areas", "skor": "Strategic score", "sisa": "left",
        "tidak_cocok": "No area matches this budget and distance yet.", "termurah": "Cheapest 70/90 house within",
        "data_area_kosong": "Area data not available. Upload harga_per_area.csv, titik_acuan.csv and ringkasan_titik_acuan.csv.",
        "Premium": "Premium", "Di atas rata-rata": "Above average", "Rata-rata": "Average", "Terjangkau": "Affordable",
        "model_judul": "Model comparison", "model_sub": "All models are tested on the same 631 test houses. Lower MAPE means more accurate.",
        "dipakai": "IN USE", "estimasi_semua": "Estimate from every model for the house in the calculator", "dari": "vs",
        "model_utama": "Main model of this version",
        "tentang_judul": "How this app works",
        "alur": [("Data", "3,553 Jabodetabek house listings (Kaggle, rumah123.com)."),
                 ("Cleaning", "Removed duplicates & unreasonable values → 3,153 houses."),
                 ("Preprocessing", "Imputation, log-transform, scaling, one-hot (14 → 33 columns)."),
                 ("Modeling", "Trained on log(price), Grid Search + 5-Fold CV."),
                 ("Evaluation", "Tested on 631 houses the model had never seen.")],
        "keterbatasan": "Limitations",
        "batas": ["Prices are listing prices, not final transaction prices.",
                  "Location uses the district/area centre, not the exact address.",
                  "Few houses in Central Jakarta and above Rp10 billion, so estimates there are less accurate.",
                  "This estimate is not an official appraisal."],
        "aturan": "Fair price rule",
        "aturan_isi": "The difference between the seller's price and the estimate is compared with the model MAPE. Within tolerance = FAIR, above = ABOVE MARKET, below = BELOW MARKET.",
        "peringatan_besar": "This house is much larger than most training data, so the estimate is less accurate.",
        "peringatan_lb": "Building size exceeds land size × floors. Please check the inputs.",
    },
}

st.set_page_config(page_title=f"{NAMA_BRAND} · Dashboard Harga Rumah", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------------
# Status bahasa & tema (disimpan di session_state agar tidak hilang saat klik)
# ---------------------------------------------------------------------
if "bahasa" not in st.session_state:
    st.session_state["bahasa"] = "ID"
if "mode_gelap" not in st.session_state:
    st.session_state["mode_gelap"] = False
if "riwayat" not in st.session_state:
    st.session_state["riwayat"] = []

BAHASA = st.session_state["bahasa"] or "ID"
T = TEKS[BAHASA]
W = TEMA["gelap" if st.session_state["mode_gelap"] else "terang"]


# ---------------------------------------------------------------------
# CSS (warna diambil dari tema aktif)
# ---------------------------------------------------------------------
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family={FONT}:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select, p, label, span, div {{ font-family: '{FONT}', sans-serif; }}
.stApp {{ background:{W['latar']}; color:{W['teks']}; }}
#MainMenu, footer {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:1320px; padding-top:.8rem; padding-bottom:2.5rem; }}
p, label, .stMarkdown, [data-testid="stWidgetLabel"] p {{ color:{W['teks']}; }}
[data-testid="stWidgetLabel"] p {{ font-size:12.5px !important; font-weight:600 !important; color:{W['redup']} !important; }}
[data-testid="stCaptionContainer"] p {{ color:{W['redup']} !important; }}
/* kartu bento: container Streamlit dengan key berawalan "bento" */
[class*="st-key-bento"] {{ background:{W['kartu']}; border:1px solid {W['garis']}; border-radius:24px; padding:20px 22px 14px; box-shadow:{W['bayang']}; }}
.kartu {{ background:{W['kartu']}; border:1px solid {W['garis']}; border-radius:24px; padding:20px 22px; box-shadow:{W['bayang']}; height:100%; }}
.kartu2 {{ background:{W['kartu2']}; border-radius:18px; padding:14px 16px; }}
.judul-kartu {{ display:flex; align-items:center; justify-content:space-between; font-size:15px; font-weight:700; color:{W['teks']}; margin-bottom:2px; }}
.sub-kartu {{ font-size:12.5px; color:{W['redup']}; margin-bottom:12px; }}
.label {{ font-size:11.5px; font-weight:600; color:{W['redup']}; text-transform:uppercase; letter-spacing:.06em; }}
.redup {{ font-size:12.5px; color:{W['redup']}; line-height:1.5; }}
.angka {{ font-size:26px; font-weight:800; letter-spacing:-.03em; color:{W['teks']}; line-height:1.15; margin-top:4px; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; font-size:11.5px; font-weight:700; padding:5px 10px; border-radius:999px; background:{W['sorot']}; color:{W['sorot_teks']}; }}
.chip-garis {{ display:inline-flex; font-size:11.5px; font-weight:600; padding:5px 10px; border-radius:999px; border:1px solid {W['garis']}; color:{W['redup']}; }}
/* navbar */
.nav {{ display:flex; align-items:center; gap:12px; height:46px; }}
.logo {{ width:38px; height:38px; border-radius:12px; background:{W['aksen']}; display:flex; align-items:center; justify-content:center; }}
.brand {{ font-weight:800; font-size:19px; letter-spacing:-.02em; color:{W['teks']}; }}
.brand span {{ color:{W['redup']}; font-weight:600; font-size:12px; display:block; letter-spacing:0; margin-top:-2px; }}
/* hero sapaan */
.sapaan h1 {{ font-size:30px; font-weight:800; letter-spacing:-.035em; color:{W['teks']}; margin:6px 0 2px; padding:0; }}
.sapaan p {{ font-size:14px; color:{W['redup']}; margin:0; }}
/* kartu hasil (gelap) */
.hasil {{ background:{W['hero']}; color:{W['hero_teks']}; border-radius:24px; padding:24px 26px; position:relative; overflow:hidden; box-shadow:{W['bayang']}; animation:muncul .45s ease; }}
.hasil .label {{ color:{W['hero_redup']}; }}
.hasil .harga {{ font-size:46px; font-weight:800; letter-spacing:-.045em; line-height:1.05; margin:10px 0 6px; color:{W['hero_teks']}; }}
.hasil .harga em {{ font-style:normal; color:{W['sorot']}; }}
.hasil .redup {{ color:{W['hero_redup']}; }}
.hasil-pita {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:14px; }}
.hasil-pita span {{ font-size:11.5px; padding:5px 10px; border-radius:999px; background:rgba(255,255,255,.09); color:{W['hero_teks']}; }}
.lingkar {{ position:absolute; right:-60px; top:-60px; width:220px; height:220px; border-radius:50%; border:28px solid {W['sorot']}; opacity:.18; }}
@keyframes muncul {{ from {{ opacity:0; transform:translateY(6px) scale(.99); }} to {{ opacity:1; transform:none; }} }}
.grid3 {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:12px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
/* status */
.pil {{ display:inline-block; padding:5px 12px; border-radius:999px; font-size:12px; font-weight:800; }}
.wajar, .Terjangkau {{ background:{W['hijau_bg']}; color:{W['hijau']}; }}
.mahal, .Premium {{ background:{W['merah_bg']}; color:{W['merah']}; }}
.murah, .Di-atas {{ background:{W['kuning_bg']}; color:{W['kuning']}; }}
.netral, .Rata-rata {{ background:{W['kartu2']}; color:{W['redup']}; }}
.skala {{ position:relative; height:10px; border-radius:999px; background:{W['kartu2']}; margin:28px 0 26px; }}
.skala-wajar {{ position:absolute; top:0; height:10px; border-radius:999px; background:{W['hijau']}; opacity:.45; }}
.penanda {{ position:absolute; top:-7px; width:4px; height:24px; border-radius:4px; transform:translateX(-50%); }}
.penanda-teks {{ position:absolute; font-size:11px; font-weight:700; white-space:nowrap; transform:translateX(-50%); color:{W['teks']}; }}
/* faktor */
.faktor {{ margin:9px 0; }}
.faktor-atas {{ display:flex; justify-content:space-between; font-size:13px; color:{W['teks']}; font-weight:600; }}
.faktor-atas b {{ font-weight:800; }}
.batang {{ height:8px; border-radius:999px; background:{W['kartu2']}; overflow:hidden; margin-top:5px; }}
.isi {{ height:100%; border-radius:999px; background:{W['aksen']}; }}
.isi.utama {{ background:{W['sorot']}; }}
/* rumah mirip */
.mirip {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }}
.rumah {{ background:{W['kartu2']}; border-radius:18px; overflow:hidden; }}
.rumah svg {{ display:block; width:100%; height:auto; }}
.rumah-isi {{ padding:12px 14px 14px; }}
.rumah-harga {{ font-size:18px; font-weight:800; color:{W['teks']}; letter-spacing:-.02em; }}
.spek {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }}
.spek span {{ font-size:11px; padding:4px 8px; border-radius:8px; background:{W['kartu']}; color:{W['redup']}; border:1px solid {W['garis']}; }}
/* tabel riwayat */
.tabel {{ width:100%; border-collapse:collapse; font-size:13px; }}
.tabel th {{ text-align:left; font-size:11.5px; font-weight:600; color:{W['redup']}; text-transform:uppercase; letter-spacing:.05em; padding:8px 10px; border-bottom:1px solid {W['garis']}; }}
.tabel td {{ padding:11px 10px; border-bottom:1px solid {W['garis']}; color:{W['teks']}; }}
/* rekomendasi area */
.rek {{ display:grid; grid-template-columns:40px 1.5fr .8fr 1.1fr 1.1fr; gap:12px; align-items:center; background:{W['kartu2']}; border-radius:16px; padding:12px 16px; margin-top:8px; }}
.rek.pertama {{ outline:2px solid {W['sorot']}; }}
.nomor {{ width:32px; height:32px; border-radius:10px; background:{W['kartu']}; display:flex; align-items:center; justify-content:center; font-weight:800; color:{W['teks']}; }}
.rek.pertama .nomor {{ background:{W['sorot']}; color:{W['sorot_teks']}; }}
.nilai {{ font-weight:800; font-size:14px; color:{W['teks']}; }}
/* widget Streamlit mengikuti tema */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {{ background:{W['input']} !important; border-color:{W['garis']} !important; border-radius:12px !important; }}
div[data-baseweb="input"] input, div[data-baseweb="select"] div, div[data-baseweb="select"] span {{ color:{W['teks']} !important; }}
[data-testid="stNumberInput"] button {{ background:{W['input']} !important; color:{W['teks']} !important; border-color:{W['garis']} !important; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {{ background:{W['kartu']} !important; color:{W['teks']} !important; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"], [data-testid="stSliderThumbValue"] {{ color:{W['teks']} !important; }}
[data-testid="stSlider"] [role="slider"] {{ background:{W['aksen']} !important; box-shadow:0 0 0 4px {W['sorot']}55 !important; }}
button[data-testid*="stBaseButton-pills"], button[data-testid*="stBaseButton-segmented_control"] {{ background:{W['kartu2']} !important; color:{W['teks']} !important; border:1px solid {W['garis']} !important; border-radius:999px !important; }}
button[data-testid="stBaseButton-pillsActive"], button[data-testid="stBaseButton-segmented_controlActive"] {{ background:{W['aksen']} !important; color:{W['aksen_teks']} !important; border-color:{W['aksen']} !important; }}
button[data-testid="stBaseButton-pillsActive"] p, button[data-testid="stBaseButton-segmented_controlActive"] p {{ color:{W['aksen_teks']} !important; }}
.stButton button {{ width:100%; border-radius:14px !important; padding:10px 18px !important; font-weight:700 !important; background:{W['aksen']} !important; color:{W['aksen_teks']} !important; border:none !important; }}
.stButton button p {{ color:{W['aksen_teks']} !important; }}
[data-testid="stToggle"] p {{ color:{W['teks']} !important; }}
/* tab sebagai menu navigasi */
.stTabs [data-baseweb="tab-list"] {{ gap:4px; background:{W['kartu']}; border:1px solid {W['garis']}; padding:5px; border-radius:16px; width:fit-content; max-width:100%; flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ height:38px; padding:0 18px; border-radius:12px; }}
.stTabs [data-baseweb="tab"] p {{ font-weight:600; color:{W['redup']}; font-size:14px; }}
.stTabs [aria-selected="true"] {{ background:{W['aksen']}; }}
.stTabs [aria-selected="true"] p {{ color:{W['aksen_teks']} !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
[data-testid="stAlert"] {{ border-radius:14px; }}
.footer {{ text-align:center; font-size:12px; color:{W['redup']}; margin-top:36px; }}
@media (max-width: 900px) {{
  .grid3, .mirip {{ grid-template-columns:1fr; }} .hasil .harga {{ font-size:36px; }}
  .rek {{ grid-template-columns:40px 1fr 1fr; }} .rek .lebar {{ grid-column:1 / -1; }}
}}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Memuat model & data
# ---------------------------------------------------------------------
@st.cache_resource
def muat_model(file):
    return joblib.load(file)


@st.cache_data
def baca_json(file):
    with open(file) as f:
        return json.load(f)


@st.cache_data
def baca_csv(file):
    return pd.read_csv(file) if os.path.exists(file) else None


@st.cache_resource
def muat_model_pembanding(nama_utama):
    hasil = {}
    for nama, file in FILE_SEMUA_MODEL.items():
        if nama != nama_utama and os.path.exists(file):
            try:
                hasil[nama] = joblib.load(file)
            except Exception:                        # library model belum terpasang -> lewati
                pass
    return hasil


@st.cache_data
def muat_perbandingan():
    hasil = baca_json("perbandingan_model.json") if os.path.exists("perbandingan_model.json") else {}
    kolom = ["mape_test_persen", "mae_test_rp", "r2_log_test", "r2_log_train"]
    for nama, file in FILE_SEMUA_INFO.items():
        if os.path.exists(file):
            isi = baca_json(file)
            hasil[nama] = {**hasil.get(nama, {}), **{k: isi[k] for k in kolom if k in isi}}
    return {n: hasil[n] for n in FILE_SEMUA_MODEL if n in hasil}


model = muat_model(FILE_MODEL_UTAMA)
info = baca_json(FILE_INFO)
FITUR = info["fitur"]
KOLOM_KATEGORI = ["city", "certificate", "property_condition", "furnishing"]
lokasi = baca_csv("lokasi_district.csv")
listing = baca_csv("data_listing.csv")                       # data asli: fitur + harga + area + latih/uji
JUMLAH_RUMAH = len(listing) if listing is not None else 3153
if listing is not None:                                      # buang 2 area yang koordinatnya salah
    listing = listing[listing["lat"].between(-6.8, -5.95) & listing["long"].between(106.35, 107.35)]
MAPE = info["mape_test_persen"] / 100


def siapkan(df):
    """Urutkan kolom seperti saat training & pastikan kolom kategori bertipe teks."""
    df = df[FITUR].copy()
    for k in KOLOM_KATEGORI:
        df[k] = df[k].astype(object)
    return df


@st.cache_data
def evaluasi_uji(nama_model):
    """Akurasi & faktor pengaruh dihitung ulang pada data uji (631 rumah) dengan model aplikasi."""
    if listing is None:
        return None
    uji = listing[listing["set"] == "uji"]
    X, y_log = siapkan(uji), np.log(uji["harga"].values)
    pred = model.predict(X)
    dalam_20 = float(np.mean(np.abs(np.exp(pred) - uji["harga"].values) / uji["harga"].values <= 0.2) * 100)
    sse_dasar = np.mean((y_log - pred) ** 2)
    grup = {"f_luas": ["land_size_m2", "building_size_m2"], "f_lokasi": ["lat", "long", "city"],
            "f_ruang": ["bedrooms", "bathrooms", "floors"], "f_fasilitas": ["electricity", "carports", "garages"],
            "f_legal": ["certificate", "property_condition", "furnishing"]}
    acak = np.random.default_rng(42)
    naik = {}
    for nama, kolom in grup.items():                         # permutation importance per kelompok faktor
        hasil = []
        for _ in range(3):
            Xa = X.copy()
            urutan = acak.permutation(len(Xa))
            Xa[kolom] = Xa[kolom].values[urutan]
            hasil.append(np.mean((y_log - model.predict(Xa)) ** 2) - sse_dasar)
        naik[nama] = max(float(np.mean(hasil)), 0.0)
    total = sum(naik.values()) or 1
    return {"dalam_20": dalam_20, "faktor": {k: v / total * 100 for k, v in naik.items()}}


# ---------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------
def rp(x):
    if BAHASA == "EN":
        return f"Rp{x / 1e9:.2f} bn" if x >= 1e9 else f"Rp{x / 1e6:,.0f} m"
    return "Rp" + f"{x / 1e9:.2f}".replace(".", ",") + " M" if x >= 1e9 else "Rp" + f"{x / 1e6:,.0f}".replace(",", ".") + " jt"


def rp_lengkap(x):
    return "Rp" + (f"{x:,.0f}" if BAHASA == "EN" else f"{x:,.0f}".replace(",", "."))


def angka(x, d=1):
    s = f"{x:,.{d}f}"
    return s if BAHASA == "EN" else s.replace(",", "_").replace(".", ",").replace("_", ".")


def html(teks):
    st.markdown(teks, unsafe_allow_html=True)


def jarak_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


WARNA_RUMAH = [("#F4C7A8", "#E07A5F"), ("#CFE3F5", "#3D5A80"), ("#E4EDC9", "#6A8D3A"), ("#F6E3B4", "#C98B2B"), ("#E5D9F2", "#7B61A8")]


def gambar_rumah(i):
    """Ilustrasi rumah sederhana (pengganti foto) dengan warna berbeda tiap kartu."""
    dinding, atap = WARNA_RUMAH[i % len(WARNA_RUMAH)]
    return (f'<svg viewBox="0 0 300 150" xmlns="http://www.w3.org/2000/svg"><rect width="300" height="150" fill="{dinding}" opacity=".45"/>'
            f'<rect x="95" y="62" width="110" height="76" rx="4" fill="#FFFFFF"/><polygon points="82,68 150,24 218,68" fill="{atap}"/>'
            f'<rect x="138" y="96" width="24" height="42" rx="12" fill="{atap}"/><rect x="106" y="80" width="22" height="20" rx="3" fill="{dinding}"/>'
            f'<rect x="172" y="80" width="22" height="20" rx="3" fill="{dinding}"/><rect x="0" y="136" width="300" height="14" fill="{atap}" opacity=".25"/>'
            f'<circle cx="250" cy="110" r="18" fill="{atap}" opacity=".35"/><rect x="248" y="112" width="4" height="26" fill="{atap}" opacity=".5"/></svg>')


IKON = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{w}" stroke-width="2.3" stroke-linecap="round" '
        'stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/></svg>')

PILIHAN_LISTRIK = [450, 900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 10600, 11000, 13200, 16500, 23000]
SERTIFIKAT = {"SHM": "SHM", "HGB": "HGB", "Lainnya" if BAHASA == "ID" else "Other": "Lainnya", "?": np.nan}
KONDISI = ({"Baru": "baru", "Bagus sekali": "bagus sekali", "Bagus": "bagus", "Renovasi": "sudah renovasi",
            "Butuh renovasi": "butuh renovasi", "?": np.nan} if BAHASA == "ID" else
           {"New": "baru", "Very good": "bagus sekali", "Good": "bagus", "Renovated": "sudah renovasi",
            "Needs work": "butuh renovasi", "?": np.nan})
PERABOT = ({"Kosong": "unfurnished", "Semi": "semi furnished", "Lengkap": "furnished", "?": np.nan} if BAHASA == "ID" else
           {"Unfurnished": "unfurnished", "Semi": "semi furnished", "Furnished": "furnished", "?": np.nan})

# ---------------------------------------------------------------------
# NAVBAR: logo · versi · bahasa · mode gelap
# ---------------------------------------------------------------------
n1, n2, n3 = st.columns([6, 1.3, 1.4], vertical_alignment="center")
with n1:
    html(f'<div class="nav"><div class="logo">{IKON.format(w=W["aksen_teks"])}</div>'
         f'<div class="brand">{NAMA_BRAND}<span>{LABEL_VERSI} · Jabodetabek</span></div></div>')
with n2:
    st.segmented_control("Bahasa", ["ID", "EN"], key="bahasa", label_visibility="collapsed")
with n3:
    st.toggle(f"🌙 {T['mode_gelap']}", key="mode_gelap")

tab_dash, tab_area, tab_model, tab_tentang = st.tabs(
    [f"📊  {T['tab_dash']}", f"📍  {T['tab_area']}", f"🤖  {T['tab_model']}", f"ℹ️  {T['tab_tentang']}"])

# =====================================================================
# TAB DASHBOARD
# =====================================================================
with tab_dash:
    html(f'<div class="sapaan"><h1>{T["sapaan"]}</h1><p>{T["sub_sapaan"]}</p></div>')
    html('<div style="height:10px"></div>')
    kiri, kanan = st.columns([4.2, 7.8], gap="medium")

    # ---------------- Kalkulator (kiri) ----------------
    with kiri:
        with st.container(key="bento_kalkulator"):
            html(f'<div class="judul-kartu">{T["kalkulator"]}<span class="chip">ML</span></div><div class="sub-kartu">{info["nama_model"]}</div>')
            daftar_kota = sorted(lokasi["city"].unique())
            kota = st.selectbox(T["kota"], daftar_kota, index=daftar_kota.index("Tangerang"), key="kota")
            daftar_area = sorted(lokasi.loc[lokasi["city"] == kota, "district"].unique())
            area_pilih = st.selectbox(T["area"], daftar_area, key=f"area_{kota}")
            titik = lokasi[(lokasi["city"] == kota) & (lokasi["district"] == area_pilih)].iloc[0]
            luas_tanah = st.slider(T["lt"], 30, 1000, 120, step=5, key="lt")
            luas_bangunan = st.slider(T["lb"], 20, 1000, 100, step=5, key="lb")
            c1, c2, c3 = st.columns(3)
            kamar_tidur = c1.number_input(T["kt"], 1, 10, 3, key="kt")
            kamar_mandi = c2.number_input(T["km"], 1, 10, 2, key="km")
            lantai = c3.number_input(T["lantai"], 1, 5, 2, key="lantai")
            c4, c5, c6 = st.columns(3)
            listrik = c4.selectbox(T["listrik"], PILIHAN_LISTRIK, index=3, key="listrik")
            carport = c5.number_input(T["carport"], 0, 10, 1, key="carport")
            garasi = c6.number_input(T["garasi"], 0, 10, 0, key="garasi")
            sertifikat = st.pills(T["sertifikat"], list(SERTIFIKAT), default="SHM", key=f"sert_{BAHASA}")
            kondisi = st.pills(T["kondisi"], list(KONDISI), default=list(KONDISI)[2], key=f"kond_{BAHASA}")
            perabot = st.pills(T["perabot"], list(PERABOT), default=list(PERABOT)[0], key=f"perabot_{BAHASA}")
            penawaran_juta = st.number_input(T["penawaran"], 0, 1_000_000, 0, step=50, key="penawaran")
            simpan = st.button(T["simpan"], key="tombol_simpan")

    # ---------------- Prediksi (berjalan otomatis setiap isian berubah) ----------------
    data_input = siapkan(pd.DataFrame([{
        "land_size_m2": luas_tanah, "building_size_m2": luas_bangunan, "bedrooms": kamar_tidur,
        "bathrooms": kamar_mandi, "carports": carport, "garages": garasi, "floors": lantai, "electricity": listrik,
        "lat": titik["lat"], "long": titik["long"], "city": kota,
        "certificate": SERTIFIKAT.get(sertifikat, np.nan), "property_condition": KONDISI.get(kondisi, np.nan),
        "furnishing": PERABOT.get(perabot, np.nan),
    }]))
    estimasi = float(np.exp(model.predict(data_input)[0]))
    bawah, atas = estimasi * (1 - MAPE), estimasi * (1 + MAPE)
    st.session_state["input_terakhir"] = data_input

    status, kelas, selisih = None, "netral", None
    if penawaran_juta > 0:
        penawaran = penawaran_juta * 1e6
        selisih = (penawaran - estimasi) / estimasi * 100
        if abs(selisih) <= info["mape_test_persen"]:
            status, kelas = "WAJAR", "wajar"
        elif selisih > 0:
            status, kelas = "MAHAL", "mahal"
        else:
            status, kelas = "MURAH", "murah"

    if simpan:
        st.session_state["riwayat"].insert(0, {
            "waktu": datetime.now().strftime("%H:%M"), "lokasi": f"{area_pilih}, {kota}",
            "ukuran": f"{luas_tanah}/{luas_bangunan} m²", "estimasi": estimasi,
            "penawaran": penawaran_juta * 1e6 if penawaran_juta else None, "status": status, "kelas": kelas})
        st.toast(T["tersimpan"])

    evaluasi = evaluasi_uji(info["nama_model"])

    # ---------------- Hasil & bento (kanan) ----------------
    with kanan:
        html(
            '<div class="hasil"><div class="lingkar"></div>'
            f'<div class="label">{T["estimasi"]}</div>'
            f'<div class="harga">{rp(estimasi).replace(" M", "<em> M</em>").replace(" bn", "<em> bn</em>").replace(" jt", "<em> jt</em>").replace(" m", "<em> m</em>")}</div>'
            f'<div class="redup">{T["kisaran"]} {rp(bawah)} – {rp(atas)}</div>'
            f'<div class="hasil-pita"><span>📍 {area_pilih}, {kota}</span><span>{luas_tanah}/{luas_bangunan} m²</span>'
            f'<span>{kamar_tidur} KT · {kamar_mandi} KM</span><span>{lantai} {T["lantai"].lower()}</span></div></div>'
        )
        teks_akurasi = (f'<div class="angka">{angka(evaluasi["dalam_20"], 0)}%</div><div class="redup">{T["akurasi_sub"]}</div>'
                        if evaluasi else f'<div class="angka">±{angka(info["mape_test_persen"])}%</div><div class="redup">{T["mape_sub"]}</div>')
        if status:
            kiri_s, kanan_s = min(bawah, penawaran) * 0.85, max(atas, penawaran) * 1.1
            pos = lambda x: (x - kiri_s) / (kanan_s - kiri_s) * 100
            kartu_cek = (
                f'<div class="kartu"><div class="label">{T["cek"]}</div><div style="margin:10px 0 6px"><span class="pil {kelas}">{T[status]}</span></div>'
                f'<div class="redup">{rp(penawaran)} · {T["selisih"]} <b>{"+" if selisih > 0 else ""}{angka(selisih)}%</b> '
                f'({T["toleransi"]} ±{angka(info["mape_test_persen"])}%). {T["saran_" + kelas]}</div>'
                f'<div class="skala"><div class="skala-wajar" style="left:{pos(bawah):.1f}%;width:{pos(atas) - pos(bawah):.1f}%"></div>'
                f'<div class="penanda" style="left:{pos(estimasi):.1f}%;background:{W["teks"]}"></div>'
                f'<div class="penanda-teks" style="left:{pos(estimasi):.1f}%;top:-22px">{T["estimasi"].split()[0]}</div>'
                f'<div class="penanda" style="left:{pos(penawaran):.1f}%;background:{W["merah"]}"></div>'
                f'<div class="penanda-teks" style="left:{pos(penawaran):.1f}%;top:20px;color:{W["merah"]}">{rp(penawaran)}</div></div>'
                f'<div class="redup" style="font-size:11.5px">{T["pita"]}</div></div>'
            )
        else:
            kartu_cek = (f'<div class="kartu"><div class="label">{T["cek"]}</div>'
                         f'<div class="redup" style="margin-top:10px">{T["isi_penawaran"]}</div></div>')
        html(
            '<div class="grid3">'
            f'<div class="kartu"><div class="label">{T["per_m2"]}</div><div class="angka">{rp_lengkap(estimasi / luas_bangunan)}</div>'
            f'<div class="redup">{rp(bawah / luas_bangunan)} – {rp(atas / luas_bangunan)}</div></div>'
            f'<div class="kartu"><div class="label">{T["akurasi"]}</div>{teks_akurasi}'
            f'<div class="redup">MAPE ±{angka(info["mape_test_persen"])}%</div></div>'
            f'{kartu_cek}</div>'
        )
        if luas_bangunan > luas_tanah * lantai:
            st.warning(T["peringatan_lb"])
        if luas_bangunan > 800 or luas_tanah > 900:
            st.warning(T["peringatan_besar"])

        html('<div style="height:12px"></div>')
        b1, b2 = st.columns([1, 1.15], gap="small")
        # Faktor pengaruh harga
        with b1:
            if evaluasi:
                urut = sorted(evaluasi["faktor"].items(), key=lambda kv: -kv[1])
                isi = "".join(
                    f'<div class="faktor"><div class="faktor-atas"><span>{T[k]}</span><b>{angka(v, 0)}%</b></div>'
                    f'<div class="batang"><div class="isi{" utama" if i == 0 else ""}" style="width:{max(v, 2):.0f}%"></div></div></div>'
                    for i, (k, v) in enumerate(urut))
                html(f'<div class="kartu"><div class="judul-kartu">{T["faktor"]}</div><div class="sub-kartu">{T["faktor_sub"]}</div>{isi}</div>')
        # Grafik median harga per kota
        with b2:
            with st.container(key="bento_grafik"):
                html(f'<div class="judul-kartu">{T["grafik"]}</div><div class="sub-kartu">{T["grafik_sub"]}</div>')
                if listing is not None:
                    med = listing.groupby("city")["harga"].median().reset_index()
                    med["harga_m"] = med["harga"] / 1e9
                    med["pilih"] = np.where(med["city"] == kota, "ya", "tidak")
                    grafik = (alt.Chart(med).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                              .encode(x=alt.X("city:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-35, labelColor=W["redup"], labelFontSize=10)),
                                      y=alt.Y("harga_m:Q", title=None, axis=alt.Axis(labelColor=W["redup"], gridColor=W["garis"], format=".1f")),
                                      color=alt.Color("pilih:N", scale=alt.Scale(domain=["ya", "tidak"], range=[W["sorot"] if st.session_state["mode_gelap"] else W["aksen"], W["grafik_biasa"]]), legend=None),
                                      tooltip=[alt.Tooltip("city:N"), alt.Tooltip("harga_m:Q", format=".2f", title="Rp miliar")])
                              .properties(height=210, background="transparent").configure_view(strokeWidth=0))
                    st.altair_chart(grafik, use_container_width=True)

    # ---------------- Baris bawah: statistik · rumah mirip · peta ----------------
    if listing is not None:
        data_kota = listing[listing["city"] == kota]
        html('<div style="height:14px"></div>')
        s1, s2, s3 = st.columns([1.1, 2.6, 1.6], gap="medium")
        with s1:
            html(
                f'<div class="kartu"><div class="judul-kartu">{T["statistik"]}</div><div class="sub-kartu">{kota}</div>'
                f'<div class="kartu2"><div class="angka" style="font-size:22px">{angka(JUMLAH_RUMAH, 0)}</div><div class="redup">{T["total"]}</div></div>'
                f'<div class="kartu2" style="margin-top:10px"><div class="angka" style="font-size:22px">{rp((data_kota["harga"] / data_kota["building_size_m2"]).median())}</div>'
                f'<div class="redup">{T["median_m2"]} {kota}</div></div>'
                f'<div class="kartu2" style="margin-top:10px"><div class="angka" style="font-size:22px">{data_kota["area"].nunique()}</div>'
                f'<div class="redup">{T["jumlah_area"]} {kota}</div></div></div>'
            )
        with s2:
            # rumah mirip: jarak gabungan ukuran (skala log), kamar, dan lokasi
            d = data_kota.copy()
            d["beda"] = (np.abs(np.log(d["land_size_m2"]) - np.log(luas_tanah)) + np.abs(np.log(d["building_size_m2"]) - np.log(luas_bangunan))
                         + 0.15 * np.abs(d["bedrooms"] - kamar_tidur) + 0.05 * jarak_km(d["lat"], d["long"], titik["lat"], titik["long"]))
            mirip = d.nsmallest(3, "beda")
            kartu = "".join(
                f'<div class="rumah">{gambar_rumah(i)}<div class="rumah-isi"><div class="rumah-harga">{rp(r.harga)}</div>'
                f'<div class="redup">{r.area} · {T["harga_iklan"]}</div><div class="spek"><span>LT {r.land_size_m2:.0f}</span>'
                f'<span>LB {r.building_size_m2:.0f}</span><span>{r.bedrooms:.0f} KT</span><span>{r.bathrooms:.0f} KM</span></div></div></div>'
                for i, r in enumerate(mirip.itertuples()))
            html(f'<div class="kartu"><div class="judul-kartu">{T["mirip"]}<span class="chip-garis">{len(data_kota)} {T["dari_iklan"]}</span></div>'
                 f'<div class="sub-kartu">{T["mirip_sub"]} {kota}</div><div class="mirip">{kartu}</div></div>')
        with s3:
            with st.container(key="bento_peta"):
                html(f'<div class="judul-kartu">{T["peta"]}</div><div class="sub-kartu">{T["peta_sub"]} {kota}</div>')
                titik_peta = data_kota.groupby(["lat", "long"]).size().reset_index(name="n")
                peta = pd.concat([
                    pd.DataFrame({"lat": titik_peta["lat"], "lon": titik_peta["long"], "warna": "#8A8F9C88",
                                  "ukuran": 60 + titik_peta["n"] * 12}),
                    pd.DataFrame({"lat": mirip["lat"], "lon": mirip["long"], "warna": W["aksen"], "ukuran": 200}),
                    pd.DataFrame({"lat": [titik["lat"]], "lon": [titik["long"]], "warna": ["#FF5A36"], "ukuran": [380]}),
                ], ignore_index=True)
                st.map(peta, latitude="lat", longitude="lon", color="warna", size="ukuran", zoom=10, height=300)

    # ---------------- Riwayat ----------------
    html('<div style="height:14px"></div>')
    with st.container(key="bento_riwayat"):
        html(f'<div class="judul-kartu">{T["riwayat"]}<span class="chip-garis">{len(st.session_state["riwayat"])}</span></div>')
        if not st.session_state["riwayat"]:
            html(f'<div class="redup" style="padding:10px 0 6px">{T["riwayat_kosong"]}</div>')
        else:
            baris = "".join(
                f'<tr><td>{r["waktu"]}</td><td>{r["lokasi"]}</td><td>{r["ukuran"]}</td><td><b>{rp(r["estimasi"])}</b></td>'
                f'<td>{rp(r["penawaran"]) if r["penawaran"] else "—"}</td>'
                f'<td><span class="pil {r["kelas"]}">{T[r["status"]] if r["status"] else T["tanpa_penawaran"]}</span></td></tr>'
                for r in st.session_state["riwayat"])
            html(f'<table class="tabel"><tr><th>{T["waktu"]}</th><th>{T["lokasi"]}</th><th>{T["ukuran"]}</th>'
                 f'<th>{T["estimasi"].split()[0]}</th><th>{T["penawaran"].split("(")[0]}</th><th>{T["status"]}</th></tr>{baris}</table>')
            if st.button(T["hapus"], key="hapus_riwayat"):
                st.session_state["riwayat"] = []
                st.rerun()

# =====================================================================
# TAB CARI AREA
# =====================================================================
with tab_area:
    area_df, titik_acuan, ringkasan = (baca_csv("harga_per_area.csv"), baca_csv("titik_acuan.csv"),
                                      baca_csv("ringkasan_titik_acuan.csv"))
    html(f'<div class="sapaan"><h1>{T["area_judul"]}</h1><p>{T["area_sub"]}</p></div><div style="height:12px"></div>')
    if area_df is None or titik_acuan is None or ringkasan is None:
        st.info(T["data_area_kosong"])
    else:
        PRIORITAS = {T["p_dekat"]: 0.7, T["p_seimbang"]: 0.5, T["p_hemat"]: 0.3}
        with st.container(key="bento_filter_area"):
            jenis = st.segmented_control(T["dekat"], [T["kampus"], T["industri"]], default=T["kampus"], key=f"jenis_{BAHASA}")
            kategori = "Industri" if jenis == T["industri"] else "Kampus"
            a1, a2, a3, a4 = st.columns([2, 1, 1, 1.2])
            tujuan = a1.selectbox(T["tujuan"], titik_acuan.loc[titik_acuan["kategori"] == kategori, "nama"].tolist(), key=f"tujuan_{kategori}")
            budget_juta = a2.number_input(T["budget"], 100, 20_000, 900, step=50, key="budget")
            jarak_maks = a3.slider(T["jarak"], 2, 25, 10, key="jarak_maks")
            prioritas = a4.select_slider(T["prioritas"], list(PRIORITAS), value=T["p_seimbang"], key=f"prio_{BAHASA}")
        budget = budget_juta * 1e6
        t = titik_acuan[titik_acuan["nama"] == tujuan].iloc[0]

        r = ringkasan[ringkasan["nama"] == tujuan]
        if not r.empty:
            r = r.iloc[0]
            kelas_harga = r["kelas_harga"].split()[0].replace("Di", "Di-atas") if r["kelas_harga"].startswith("Di") else r["kelas_harga"]
            iklan = rp(r["median_harga_iklan"]) if pd.notna(r["median_harga_iklan"]) else "-"
            html(
                '<div class="grid3" style="grid-template-columns:1.4fr 1fr 1fr">'
                f'<div class="hasil"><div class="lingkar"></div><div class="label">{T["sekitar"]} {tujuan} · 5 km</div>'
                f'<div class="harga" style="font-size:38px">{rp(r["harga_standar_median"])}</div>'
                f'<span class="pil {kelas_harga}">{T.get(r["kelas_harga"], r["kelas_harga"])}</span>'
                f'<div class="redup" style="margin-top:8px">{T["indeks"]} {int(round(r["indeks_harga"]))} · {T["indeks_sub"]}</div></div>'
                f'<div class="kartu"><div class="label">{T["iklan_sekitar"]}</div><div class="angka">{iklan}</div>'
                f'<div class="redup">{int(r["jumlah_iklan"])} {T["dari_iklan"]}</div></div>'
                f'<div class="kartu"><div class="label">{T["rentang"]}</div><div class="angka" style="font-size:20px">{rp(r["harga_standar_min"])} – {rp(r["harga_standar_max"])}</div>'
                f'<div class="redup">{int(r["jumlah_area"])} area · 5 km</div></div></div>'
            )

        hasil = area_df.copy()
        hasil["jarak_km"] = jarak_km(hasil["lat"].values, hasil["long"].values, t["lat"], t["long"])
        radius = hasil[hasil["jarak_km"] <= jarak_maks]
        cocok = radius[(radius["harga_standar"] <= budget) & radius["data_cukup"]].copy()
        bobot = PRIORITAS[prioritas]
        cocok["skor"] = 100 * (bobot * (1 - cocok["jarak_km"] / jarak_maks) + (1 - bobot) * (1 - cocok["harga_standar"] / budget))
        cocok = cocok.sort_values("skor", ascending=False).head(5)

        html('<div style="height:14px"></div>')
        k1, k2 = st.columns([1.35, 1], gap="medium")
        with k1:
            with st.container(key="bento_rekomendasi"):
                html(f'<div class="judul-kartu">{len(cocok)} {T["rekomendasi"]}</div><div class="sub-kartu">{tujuan}</div>')
                if cocok.empty:
                    termurah = radius.loc[radius["data_cukup"], "harga_standar"]
                    st.info(T["tidak_cocok"] + (f' {T["termurah"]} {jarak_maks} km: {rp(termurah.min())}.' if not termurah.empty else ""))
                else:
                    html("".join(
                        f'<div class="rek{" pertama" if i == 1 else ""}"><div class="nomor">{i}</div>'
                        f'<div class="lebar"><div class="nilai">{b.area}</div><div class="redup">{b.city} · {b.jumlah_iklan} {T["dari_iklan"]}</div></div>'
                        f'<div><div class="redup">km</div><div class="nilai">{angka(b.jarak_km)}</div></div>'
                        f'<div><div class="redup">70/90</div><div class="nilai">{rp(b.harga_standar)}</div><div class="redup">{T["sisa"]} {rp(budget - b.harga_standar)}</div></div>'
                        f'<div><div class="redup">{T["skor"]} <b>{b.skor:.0f}</b></div><div class="batang"><div class="isi{" utama" if i == 1 else ""}" style="width:{max(b.skor, 3):.0f}%"></div></div></div></div>'
                        for i, b in enumerate(cocok.itertuples(), start=1)))
        with k2:
            with st.container(key="bento_peta_area"):
                html(f'<div class="judul-kartu">{T["peta"]}</div><div class="sub-kartu">{tujuan} · {jarak_maks} km</div>')
                peta = pd.concat([
                    pd.DataFrame({"lat": radius["lat"], "lon": radius["long"], "warna": "#8A8F9C88", "ukuran": 110}),
                    pd.DataFrame({"lat": cocok["lat"], "lon": cocok["long"], "warna": W["aksen"], "ukuran": 260}),
                    pd.DataFrame({"lat": [t["lat"]], "lon": [t["long"]], "warna": ["#FF5A36"], "ukuran": [420]}),
                ], ignore_index=True)
                st.map(peta, latitude="lat", longitude="lon", color="warna", size="ukuran", zoom=11, height=380)

# =====================================================================
# TAB MODEL
# =====================================================================
with tab_model:
    perbandingan = muat_perbandingan()
    pembanding = muat_model_pembanding(info["nama_model"])
    html(f'<div class="sapaan"><h1>{T["model_judul"]}</h1><p>{T["model_sub"]}</p></div><div style="height:12px"></div>')
    if perbandingan:
        mape_maks = max(v.get("mape_test_persen", 0) for v in perbandingan.values()) or 1
        terbaik = min(perbandingan, key=lambda n: perbandingan[n].get("mape_test_persen", 999))
        kartu = ""
        for nama, m in perbandingan.items():
            dipakai = nama == info["nama_model"]
            baris = "".join(
                f'<div class="faktor-atas" style="font-weight:500;padding:6px 0;border-top:1px solid {W["garis"]}"><span class="redup">{lbl}</span><b>{val}</b></div>'
                for lbl, val in [("MAE", rp(m["mae_test_rp"]) if "mae_test_rp" in m else "-"),
                                 ("R² uji (log)" if BAHASA == "ID" else "R² test (log)", angka(m["r2_log_test"], 3) if "r2_log_test" in m else "-"),
                                 ("R² latih (log)" if BAHASA == "ID" else "R² train (log)", angka(m["r2_log_train"], 3) if pd.notna(m.get("r2_log_train")) else "-")])
            kartu += (f'<div class="kartu" style="{"outline:2px solid " + W["aksen"] if dipakai else ""}">'
                      f'<div class="judul-kartu">{nama}{"<span class=chip>" + T["dipakai"] + "</span>" if dipakai else ""}</div>'
                      f'<div class="label" style="margin-top:8px">MAPE</div><div class="angka">{angka(m.get("mape_test_persen", 0), 2)}%</div>'
                      f'<div class="batang" style="margin:8px 0 10px"><div class="isi{" utama" if nama == terbaik else ""}" style="width:{m.get("mape_test_persen", 0) / mape_maks * 100:.0f}%"></div></div>{baris}</div>')
        html(f'<div class="grid3" style="grid-template-columns:repeat(auto-fit, minmax(230px, 1fr))">{kartu}</div>')

    html(f'<div class="judul-kartu" style="margin:22px 0 10px;font-size:17px">{T["estimasi_semua"]}</div>')
    data_input = st.session_state["input_terakhir"]
    semua = {n: (model if n == info["nama_model"] else pembanding[n]) for n in FILE_SEMUA_MODEL
             if n == info["nama_model"] or n in pembanding}
    est_utama = float(np.exp(model.predict(data_input)[0]))
    kartu = ""
    for nama, m in semua.items():
        est = float(np.exp(m.predict(data_input)[0]))
        beda = (est - est_utama) / est_utama * 100
        ket = T["model_utama"] if nama == info["nama_model"] else f'{"+" if beda >= 0 else ""}{angka(beda)}% {T["dari"]} {info["nama_model"]}'
        kartu += f'<div class="kartu"><div class="label">{nama}</div><div class="angka">{rp(est)}</div><div class="redup">{ket}</div></div>'
    html(f'<div class="grid3" style="grid-template-columns:repeat(auto-fit, minmax(200px, 1fr))">{kartu}</div>')

# =====================================================================
# TAB TENTANG
# =====================================================================
with tab_tentang:
    html(f'<div class="sapaan"><h1>{T["tentang_judul"]}</h1></div><div style="height:12px"></div>')
    html('<div class="grid3" style="grid-template-columns:repeat(auto-fit, minmax(180px, 1fr))">' + "".join(
        f'<div class="kartu"><span class="chip">0{i}</span><div class="judul-kartu" style="margin-top:10px">{j}</div><div class="redup">{d}</div></div>'
        for i, (j, d) in enumerate(T["alur"], start=1)) + '</div>')
    html('<div style="height:14px"></div>')
    html('<div class="grid2">'
         f'<div class="kartu"><div class="judul-kartu">{T["aturan"]}</div><div class="redup" style="margin-top:6px">{T["aturan_isi"]}</div>'
         f'<div style="display:flex;gap:6px;margin-top:12px;flex-wrap:wrap"><span class="pil wajar">{T["WAJAR"]}</span>'
         f'<span class="pil mahal">{T["MAHAL"]}</span><span class="pil murah">{T["MURAH"]}</span></div></div>'
         f'<div class="kartu"><div class="judul-kartu">{T["keterbatasan"]}</div>'
         + "".join(f'<div class="redup" style="margin-top:6px">• {b}</div>' for b in T["batas"]) + '</div></div>')

html(f'<div class="footer">{NAMA_BRAND} · Proyek Akhir Pembelajaran Mesin · {NAMA_TIM}</div>')
