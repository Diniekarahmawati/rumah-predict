# =====================================================================
# RumahWajar — Estimasi, Cek Kewajaran & Cari Area Rumah Jabodetabek
# Proyek Akhir Pembelajaran Mesin
#
# Halaman dibagi menjadi 4 tab:
#   1. Cek Harga         : input spesifikasi -> estimasi harga -> cek kewajaran
#   2. Cari Area         : rekomendasi area dekat kampus / kawasan industri (notebook 04)
#   3. Perbandingan Model: metrik & estimasi dari semua model yang tersedia
#   4. Tentang           : data, metode, keterbatasan, tim
#
# File yang dibutuhkan di folder yang sama:
#   model_terbaik.joblib, info_model.json   : model utama (notebook 03)
#   lokasi_district.csv                     : area -> koordinat (notebook 02)
# File opsional:
#   model_*.joblib, info_model_*.json       : model pembanding
#   harga_per_area.csv, titik_acuan.csv,
#   ringkasan_titik_acuan.csv               : data tab "Cari Area" (notebook 04)
# =====================================================================

import json                      # membaca file info model
import os                        # mengecek apakah file opsional ada

import joblib                    # memuat model yang sudah dilatih
import numpy as np               # np.exp untuk mengembalikan log -> Rupiah
import pandas as pd              # menyusun input menjadi tabel
import streamlit as st           # membuat tampilan web

# =====================================================================
# PENGATURAN MODEL  ->  model utama yang dipakai versi aplikasi ini
# =====================================================================
FILE_MODEL_UTAMA = "model_xgboost.joblib"   # file model utama
FILE_INFO = "info_model_xgboost.json"   # file info/metrik model utama
LABEL_VERSI = "Versi XGBoost"           # label kecil di bagian atas halaman

# Semua model yang bisa ditampilkan di tab perbandingan (file yang belum ada akan dilewati)
FILE_SEMUA_MODEL = {
    "Linear Regression": "model_linear_regression.joblib",
    "Random Forest": "model_random_forest.joblib",
    "Gradient Boosting": "model_terbaik.joblib",
    "XGBoost": "model_xgboost.joblib",
}
# File info (metrik) tiap model
FILE_SEMUA_INFO = {
    "Linear Regression": "info_model_linear_regression.json",
    "Random Forest": "info_model_random_forest.json",
    "Gradient Boosting": "info_model.json",
    "XGBoost": "info_model_xgboost.json",
}

# =====================================================================
# PENGATURAN TAMPILAN  ->  ubah nilai di bagian ini untuk mengganti look
# =====================================================================
NAMA_BRAND = "RumahWajar"
FONT = "Plus Jakarta Sans"                        # font dari fonts.google.com
WARNA_UTAMA = "#0E6B63"                           # hijau-biru (tombol, aksen)
WARNA_UTAMA_GELAP = "#0A4F49"                     # versi gelap (hero)
WARNA_UTAMA_MUDA = "#E3F1EE"                      # latar lembut aksen
WARNA_AKSEN = "#F2A541"                           # oranye hangat (penanda)
WARNA_LATAR = "#F4F6F5"                           # latar halaman
WARNA_KARTU = "#FFFFFF"                           # latar kartu
WARNA_TEKS = "#14262F"                            # teks utama (biru gelap)
WARNA_TEKS_REDUP = "#5E6E73"                      # teks keterangan
WARNA_GARIS = "#E2E8E6"                           # garis tipis
SUDUT = "18px"                                    # kelengkungan sudut kartu
NAMA_TIM = "Dini Eka Rahmawati · Siti Lativa Irmiyati · Lukiman Hakim"
# =====================================================================

st.set_page_config(page_title=f"{NAMA_BRAND} · Harga Rumah Jabodetabek", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------------
# CSS: semua gaya tampilan ada di sini (memakai pengaturan di atas)
# ---------------------------------------------------------------------
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family={FONT.replace(" ", "+")}:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select, p, label {{ font-family: '{FONT}', sans-serif !important; }}
.stApp {{ background: {WARNA_LATAR}; color: {WARNA_TEKS}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1200px; padding-top: 1rem; padding-bottom: 3rem; }}
/* navbar */
.nav {{ display:flex; align-items:center; justify-content:space-between; padding:10px 4px 14px; border-bottom:1px solid {WARNA_GARIS}; }}
.brand {{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:20px; color:{WARNA_TEKS}; letter-spacing:-0.02em; }}
.brand-ikon {{ width:36px; height:36px; border-radius:10px; background:{WARNA_UTAMA}; display:flex; align-items:center; justify-content:center; }}
.nav-kanan {{ display:flex; gap:10px; align-items:center; font-size:13px; color:{WARNA_TEKS_REDUP}; }}
.lencana {{ display:inline-block; font-size:12px; font-weight:700; padding:6px 12px; border-radius:999px; background:{WARNA_UTAMA_MUDA}; color:{WARNA_UTAMA_GELAP}; }}
/* hero */
.hero {{ position:relative; overflow:hidden; border-radius:24px; margin-top:18px; padding:44px 48px; color:#FFFFFF;
        background: linear-gradient(120deg, {WARNA_UTAMA_GELAP} 0%, {WARNA_UTAMA} 60%, #1E8A7F 100%); }}
.hero-isi {{ position:relative; z-index:2; max-width:640px; }}
.hero-label {{ display:inline-block; font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,.14); }}
.hero h1 {{ font-size:44px; line-height:1.08; font-weight:800; letter-spacing:-0.03em; margin:16px 0 12px; color:#FFFFFF; padding:0; }}
.hero p {{ font-size:16px; line-height:1.6; color:rgba(255,255,255,.85); margin:0; }}
.hero-angka {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:26px; }}
.hero-angka div {{ background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18); border-radius:14px; padding:12px 18px; }}
.hero-angka b {{ display:block; font-size:22px; font-weight:800; }}
.hero-angka span {{ font-size:12px; color:rgba(255,255,255,.8); }}
.hero-gambar {{ position:absolute; right:-10px; bottom:-6px; width:470px; opacity:.95; z-index:1; }}
/* tab */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{WARNA_KARTU}; padding:6px; border-radius:14px; margin-top:22px; box-shadow:0 1px 3px rgba(20,38,47,.06); }}
.stTabs [data-baseweb="tab"] {{ height:44px; padding:0 20px; border-radius:10px; font-weight:600; color:{WARNA_TEKS_REDUP}; }}
.stTabs [aria-selected="true"] {{ background:{WARNA_UTAMA}; color:#FFFFFF !important; }}
.stTabs [aria-selected="true"] p {{ color:#FFFFFF !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
/* kartu umum */
.kartu {{ background:{WARNA_KARTU}; border:1px solid {WARNA_GARIS}; border-radius:{SUDUT}; padding:22px 24px; }}
.judul-bagian {{ font-size:22px; font-weight:800; letter-spacing:-0.02em; color:{WARNA_TEKS}; margin:18px 0 4px; }}
.sub-bagian {{ font-size:14px; color:{WARNA_TEKS_REDUP}; margin-bottom:14px; line-height:1.55; }}
.label-kecil {{ font-size:12px; font-weight:600; color:{WARNA_TEKS_REDUP}; text-transform:uppercase; letter-spacing:.06em; }}
.teks-redup {{ font-size:13px; color:{WARNA_TEKS_REDUP}; line-height:1.55; }}
.angka-besar {{ font-size:36px; font-weight:800; letter-spacing:-0.03em; color:{WARNA_TEKS}; margin:6px 0 2px; }}
.angka-sedang {{ font-size:22px; font-weight:800; letter-spacing:-0.02em; color:{WARNA_TEKS}; margin:6px 0 2px; }}
/* form (container dengan key) */
.st-key-kartu_form, .st-key-kartu_area {{ background:{WARNA_KARTU}; border:1px solid {WARNA_GARIS}; border-radius:{SUDUT}; padding:22px 24px 10px; }}
.st-key-kartu_form [data-testid="stForm"] {{ border:none; padding:0; }}
div[data-baseweb="input"], div[data-baseweb="select"] > div {{ border-radius:10px !important; background:#F7F9F8 !important; border-color:{WARNA_GARIS} !important; }}
[data-testid="stFormSubmitButton"] button {{ width:100%; border-radius:12px !important; padding:12px 22px !important; font-weight:700 !important; background:{WARNA_UTAMA} !important; border:none !important; color:#FFFFFF !important; }}
[data-testid="stFormSubmitButton"] button:hover {{ background:{WARNA_UTAMA_GELAP} !important; }}
[data-testid="stWidgetLabel"] p {{ font-size:13px !important; font-weight:600 !important; color:{WARNA_TEKS} !important; }}
.judul-grup {{ font-size:13px; font-weight:700; color:{WARNA_UTAMA}; text-transform:uppercase; letter-spacing:.06em; margin:10px 0 2px; }}
/* hasil cek harga */
.hasil-utama {{ background:{WARNA_UTAMA_GELAP}; color:#FFFFFF; border-radius:{SUDUT}; padding:24px 26px; }}
.hasil-utama .label-kecil {{ color:rgba(255,255,255,.75); }}
.hasil-utama .angka-besar {{ color:#FFFFFF; font-size:40px; }}
.hasil-utama .teks-redup {{ color:rgba(255,255,255,.8); }}
.spesifikasi {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:14px; }}
.spesifikasi span {{ font-size:12px; padding:5px 10px; border-radius:999px; background:rgba(255,255,255,.14); }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; }}
.pil {{ display:inline-block; padding:6px 14px; border-radius:999px; font-size:13px; font-weight:800; margin:8px 0 6px; }}
.wajar, .terjangkau {{ background:#DDF2E4; color:#1B6B36; }}
.mahal, .premium {{ background:#FBE1DA; color:#9A2E16; }}
.murah, .atas {{ background:#FCEFD3; color:#7A5300; }}
.rata {{ background:#E8EDF0; color:{WARNA_TEKS}; }}
.skala {{ position:relative; height:12px; border-radius:999px; background:#EDF1F0; margin:30px 0 30px; }}
.skala-wajar {{ position:absolute; top:0; height:12px; border-radius:999px; background:#BFE3D0; }}
.penanda {{ position:absolute; top:-7px; width:4px; height:26px; border-radius:4px; transform:translateX(-50%); }}
.penanda-teks {{ position:absolute; font-size:11px; font-weight:700; white-space:nowrap; transform:translateX(-50%); }}
.kosong {{ background:{WARNA_KARTU}; border:1px dashed #C9D3D0; border-radius:{SUDUT}; padding:44px 30px; text-align:center; }}
.kosong b {{ display:block; font-size:18px; margin:10px 0 6px; color:{WARNA_TEKS}; }}
/* perbandingan model */
.grid-model {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px; }}
.kartu-dipakai {{ border:2px solid {WARNA_UTAMA}; }}
.lencana-kecil {{ display:inline-block; font-size:10px; font-weight:800; padding:3px 8px; border-radius:999px; background:{WARNA_UTAMA}; color:#FFFFFF; margin-left:6px; vertical-align:middle; letter-spacing:.04em; }}
.nama-model {{ font-size:15px; font-weight:800; color:{WARNA_TEKS}; }}
.baris-metrik {{ display:flex; justify-content:space-between; font-size:13px; padding:6px 0; border-top:1px solid {WARNA_GARIS}; color:{WARNA_TEKS_REDUP}; }}
.baris-metrik b {{ color:{WARNA_TEKS}; }}
.batang {{ height:8px; border-radius:999px; background:#EDF1F0; overflow:hidden; margin:6px 0 10px; }}
.isi-batang {{ height:100%; background:{WARNA_UTAMA}; border-radius:999px; }}
/* cari area */
.kartu-rek {{ display:grid; grid-template-columns:44px 1.6fr 0.8fr 1.1fr 1.2fr; gap:14px; align-items:center; background:{WARNA_KARTU}; border:1px solid {WARNA_GARIS}; border-radius:16px; padding:14px 20px; margin-top:10px; }}
.nomor {{ width:36px; height:36px; border-radius:10px; background:{WARNA_UTAMA_MUDA}; color:{WARNA_UTAMA_GELAP}; display:flex; align-items:center; justify-content:center; font-weight:800; }}
.rek-pertama .nomor {{ background:{WARNA_UTAMA}; color:#FFFFFF; }}
.nama-area {{ font-weight:700; font-size:15px; color:{WARNA_TEKS}; }}
.nilai {{ font-weight:700; font-size:15px; color:{WARNA_TEKS}; }}
.grid-3 {{ display:grid; grid-template-columns:1.3fr 1fr 1fr; gap:14px; }}
/* tentang */
.alur {{ display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; }}
.alur div {{ background:{WARNA_KARTU}; border:1px solid {WARNA_GARIS}; border-radius:14px; padding:14px; font-size:13px; color:{WARNA_TEKS_REDUP}; }}
.alur b {{ display:block; color:{WARNA_TEKS}; font-size:14px; margin-bottom:4px; }}
.alur i {{ font-style:normal; color:{WARNA_UTAMA}; font-weight:800; font-size:12px; }}
.footer {{ text-align:center; font-size:12px; color:{WARNA_TEKS_REDUP}; margin-top:44px; padding-top:18px; border-top:1px solid {WARNA_GARIS}; }}
@media (max-width: 900px) {{
  .hero {{ padding:30px 24px 150px; }} .hero h1 {{ font-size:32px; }} .hero-gambar {{ width:320px; }}
  .nav-kanan span.sembunyi {{ display:none; }}
  .grid-2, .grid-3 {{ grid-template-columns:1fr; }} .alur {{ grid-template-columns:1fr 1fr; }}
  .kartu-rek {{ grid-template-columns:44px 1fr 1fr; row-gap:10px; }} .rek-nama {{ grid-column:2 / -1; }}
  .rek-jarak {{ grid-column:2; }} .rek-skor {{ grid-column:1 / -1; }}
}}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Memuat model & data (disimpan di cache agar tidak dimuat ulang tiap klik)
# ---------------------------------------------------------------------
@st.cache_resource
def muat_model():
    return joblib.load(FILE_MODEL_UTAMA)                 # pipeline: preprocessing + model utama


@st.cache_data
def muat_info():
    with open(FILE_INFO) as f:
        return json.load(f)                              # nama model, MAPE, MAE, daftar fitur


@st.cache_data
def muat_lokasi():
    return pd.read_csv("lokasi_district.csv")            # kolom: city, district, lat, long


@st.cache_resource
def muat_model_pembanding(nama_utama):
    hasil = {}
    for nama, file in FILE_SEMUA_MODEL.items():
        if nama != nama_utama and os.path.exists(file):  # lewati model utama & file yang belum ada
            try:
                hasil[nama] = joblib.load(file)
            except Exception:                            # mis. library model belum terpasang -> lewati
                pass
    return hasil


@st.cache_data
def muat_perbandingan():
    hasil = {}
    if os.path.exists("perbandingan_model.json"):        # metrik dari notebook 03 (jika ada)
        with open("perbandingan_model.json") as f:
            hasil = json.load(f)
    kolom = ["mape_test_persen", "mae_test_rp", "r2_log_test", "r2_log_train"]
    for nama, file in FILE_SEMUA_INFO.items():           # lengkapi dari file info tiap versi
        if os.path.exists(file):
            with open(file) as f:
                isi = json.load(f)
            hasil[nama] = {**hasil.get(nama, {}), **{k: isi[k] for k in kolom if k in isi}}   # gabung, file info lebih baru
    urutan = [n for n in FILE_SEMUA_MODEL if n in hasil] # urutkan: LR, RF, GB, XGBoost
    return {n: hasil[n] for n in urutan}


@st.cache_data
def muat_data_area():
    file_area = ["harga_per_area.csv", "titik_acuan.csv", "ringkasan_titik_acuan.csv"]
    if not all(os.path.exists(f) for f in file_area):
        return None                                      # tab Cari Area akan menampilkan pesan
    return tuple(pd.read_csv(f) for f in file_area)


model = muat_model()
info = muat_info()
lokasi = muat_lokasi()
model_pembanding = muat_model_pembanding(info["nama_model"])
perbandingan = muat_perbandingan()
data_area = muat_data_area()

MAPE = info["mape_test_persen"] / 100                    # contoh: 22.95% -> 0.2295, batas toleransi


# ---------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------
def format_rp(x):
    """Rupiah singkat: Rp1,81 miliar / Rp850 juta."""
    if x >= 1e9:
        return "Rp" + f"{x / 1e9:.2f}".replace(".", ",") + " miliar"
    return "Rp" + f"{x / 1e6:,.0f}".replace(",", ".") + " juta"


def format_rp_lengkap(x):
    """Rupiah lengkap: Rp1.805.698.325."""
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def koma(x, desimal=1):
    """Angka desimal gaya Indonesia: 22.95 -> 22,9."""
    return f"{x:.{desimal}f}".replace(".", ",")


def html(teks):
    """Tampilkan potongan HTML (satu baris, tanpa baris kosong)."""
    st.markdown(teks, unsafe_allow_html=True)


def jarak_km(lat1, lon1, lat2, lon2):
    """Jarak garis lurus (rumus Haversine) dalam km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


IKON_RUMAH = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/>'
              '<path d="M10 21v-6h4v6"/></svg>')

# Ilustrasi deretan rumah (SVG buatan sendiri, putih transparan agar menyatu dengan hero)
GAMBAR_HERO = (
    '<svg class="hero-gambar" viewBox="0 0 470 230" xmlns="http://www.w3.org/2000/svg">'
    '<g fill="#FFFFFF">'
    '<rect x="20" y="120" width="110" height="110" opacity=".10"/><polygon points="10,124 75,70 140,124" opacity=".14"/>'
    '<rect x="48" y="160" width="26" height="70" opacity=".18"/><rect x="88" y="146" width="28" height="24" opacity=".22"/>'
    '<rect x="150" y="90" width="150" height="140" opacity=".16"/><polygon points="136,96 225,30 314,96" opacity=".22"/>'
    '<rect x="172" y="120" width="40" height="34" opacity=".30"/><rect x="238" y="120" width="40" height="34" opacity=".30"/>'
    '<rect x="206" y="172" width="38" height="58" opacity=".34"/>'
    '<rect x="320" y="134" width="120" height="96" opacity=".10"/><polygon points="310,138 380,88 450,138" opacity=".14"/>'
    '<rect x="340" y="160" width="30" height="26" opacity=".2"/><rect x="392" y="170" width="26" height="60" opacity=".2"/>'
    '</g><rect x="0" y="226" width="470" height="4" fill="#FFFFFF" opacity=".25"/>'
    f'<circle cx="420" cy="40" r="16" fill="{WARNA_AKSEN}" opacity=".9"/>'
    '</svg>'
)

# Pilihan isian (sama dengan kategori yang dikenal model saat training)
PILIHAN_LISTRIK = [450, 900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 10600, 11000, 13200, 16500, 23000]
PILIHAN_SERTIFIKAT = {"SHM (Sertifikat Hak Milik)": "SHM", "HGB (Hak Guna Bangunan)": "HGB",
                      "Lainnya (PPJB, girik, adat, dll.)": "Lainnya", "Tidak tahu": np.nan}
PILIHAN_KONDISI = {"Baru": "baru", "Bagus sekali": "bagus sekali", "Bagus": "bagus",
                   "Sudah renovasi": "sudah renovasi", "Butuh renovasi": "butuh renovasi", "Tidak tahu": np.nan}
PILIHAN_PERABOT = {"Unfurnished (kosong)": "unfurnished", "Semi furnished": "semi furnished",
                   "Furnished (lengkap)": "furnished", "Tidak tahu": np.nan}

# ---------------------------------------------------------------------
# NAVBAR + HERO
# ---------------------------------------------------------------------
jumlah_rumah = f"{info['jumlah_data_train'] + 631:,}".replace(",", ".")      # 2.522 latih + 631 uji
html(
    f'<div class="nav"><div class="brand"><span class="brand-ikon">{IKON_RUMAH}</span>{NAMA_BRAND}</div>'
    '<div class="nav-kanan"><span class="sembunyi">Proyek Akhir Pembelajaran Mesin</span>'
    f'<span class="lencana">{LABEL_VERSI}</span></div></div>'
)
html(
    '<div class="hero"><div class="hero-isi">'
    '<span class="hero-label">Jabodetabek · Machine Learning</span>'
    '<h1>Tahu harga wajar rumah sebelum menawar</h1>'
    '<p>Estimasi harga pasar dari spesifikasi rumah, cek apakah harga penjual wajar, '
    'dan temukan area terjangkau di dekat kampus atau kawasan industri.</p>'
    '<div class="hero-angka">'
    f'<div><b>{jumlah_rumah}</b><span>listing rumah dianalisis</span></div>'
    '<div><b>9</b><span>kota/kabupaten</span></div>'
    f'<div><b>±{koma(info["mape_test_persen"])}%</b><span>rata-rata kesalahan (MAPE)</span></div>'
    f'</div></div>{GAMBAR_HERO}</div>'
)

tab_cek, tab_area, tab_model, tab_tentang = st.tabs(
    ["🏠  Cek Harga", "📍  Cari Area", "📊  Perbandingan Model", "ℹ️  Tentang"])

# =====================================================================
# TAB 1 — CEK HARGA
# =====================================================================
with tab_cek:
    html('<div class="judul-bagian">Cek harga rumah incaranmu</div>'
         '<div class="sub-bagian">Isi spesifikasi rumah, lalu (opsional) masukkan harga dari penjual '
         'untuk melihat apakah harganya wajar.</div>')
    kiri, kanan = st.columns([1.15, 1], gap="large")

    with kiri:
        with st.container(key="kartu_form"):
            html('<div class="judul-grup">📍 Lokasi</div>')
            daftar_kota = sorted(lokasi["city"].unique())
            kota = st.pills("Kota/Kabupaten", daftar_kota, selection_mode="single", default=daftar_kota[0])
            if kota is None:                                   # jika pil dibatalkan, pakai kota pertama
                kota = daftar_kota[0]
            daftar_kecamatan = sorted(lokasi.loc[lokasi["city"] == kota, "district"].unique())
            kecamatan = st.selectbox("Kecamatan/Area", daftar_kecamatan)
            titik = lokasi[(lokasi["city"] == kota) & (lokasi["district"] == kecamatan)].iloc[0]

            with st.form("form_rumah", border=False):
                html('<div class="judul-grup">📐 Ukuran & ruangan</div>')
                k1, k2 = st.columns(2)
                with k1:
                    luas_tanah = st.number_input("Luas tanah (m²)", min_value=15, max_value=10000, value=120, step=5)
                    kamar_tidur = st.number_input("Kamar tidur", min_value=1, max_value=10, value=3)
                    lantai = st.number_input("Jumlah lantai", min_value=1, max_value=5, value=2)
                with k2:
                    luas_bangunan = st.number_input("Luas bangunan (m²)", min_value=20, max_value=10000, value=100, step=5)
                    kamar_mandi = st.number_input("Kamar mandi", min_value=1, max_value=10, value=2)
                    listrik = st.selectbox("Daya listrik (VA)", PILIHAN_LISTRIK, index=PILIHAN_LISTRIK.index(2200))

                html('<div class="judul-grup">🚗 Parkir</div>')
                k3, k4 = st.columns(2)
                with k3:
                    carport = st.number_input("Carport", min_value=0, max_value=10, value=1)
                with k4:
                    garasi = st.number_input("Garasi", min_value=0, max_value=10, value=0)

                html('<div class="judul-grup">📄 Legalitas & kondisi</div>')
                k5, k6, k7 = st.columns(3)
                with k5:
                    sertifikat = st.selectbox("Sertifikat", list(PILIHAN_SERTIFIKAT))
                with k6:
                    kondisi = st.selectbox("Kondisi", list(PILIHAN_KONDISI), index=2)
                with k7:
                    perabot = st.selectbox("Perabot", list(PILIHAN_PERABOT))

                html('<div class="judul-grup">💰 Harga dari penjual (opsional)</div>')
                penawaran_juta = st.number_input(
                    "Harga penawaran (juta Rp)", min_value=0, max_value=1_000_000, value=0, step=50,
                    help="Contoh: 1500 = Rp1,5 miliar. Isi 0 jika hanya ingin estimasi.")
                tombol = st.form_submit_button("Hitung estimasi harga  →", type="primary")

    with kanan:
        if not tombol and "input_terakhir" not in st.session_state:
            html('<div class="kosong"><div style="font-size:40px">🏡</div>'
                 '<b>Hasil estimasi muncul di sini</b>'
                 '<div class="teks-redup">Pilih lokasi dan isi spesifikasi rumah di sebelah kiri, '
                 'lalu tekan <b>Hitung estimasi harga</b>.</div></div>')

        if tombol:
            # 1) Susun input menjadi 1 baris tabel dengan nama kolom yang sama seperti saat training
            data_input = pd.DataFrame([{
                "land_size_m2": luas_tanah, "building_size_m2": luas_bangunan,
                "bedrooms": kamar_tidur, "bathrooms": kamar_mandi,
                "carports": carport, "garages": garasi, "floors": lantai, "electricity": listrik,
                "lat": titik["lat"], "long": titik["long"], "city": kota,
                "certificate": PILIHAN_SERTIFIKAT[sertifikat],      # NaN -> "Tidak diketahui" oleh pipeline
                "property_condition": PILIHAN_KONDISI[kondisi],
                "furnishing": PILIHAN_PERABOT[perabot],
            }])[info["fitur"]]                                     # urutkan kolom seperti saat training
            for kolom in ["city", "certificate", "property_condition", "furnishing"]:
                data_input[kolom] = data_input[kolom].astype(object)   # kolom kategori bertipe teks

            # 2) Prediksi: model menghasilkan log(harga), dikembalikan ke Rupiah dengan exp
            estimasi = float(np.exp(model.predict(data_input)[0]))
            st.session_state["input_terakhir"] = data_input       # dipakai di tab Perbandingan Model
            st.session_state["ringkasan_input"] = (f"{kecamatan}, {kota} · LT {luas_tanah} m² · LB {luas_bangunan} m² · "
                                                   f"{kamar_tidur} KT · {kamar_mandi} KM")
            batas_bawah, batas_atas = estimasi * (1 - MAPE), estimasi * (1 + MAPE)

            html(
                '<div class="hasil-utama"><div class="label-kecil">Estimasi harga pasar</div>'
                f'<div class="angka-besar">{format_rp(estimasi)}</div>'
                f'<div class="teks-redup">Kisaran wajar {format_rp(batas_bawah)} – {format_rp(batas_atas)}</div>'
                f'<div class="spesifikasi"><span>📍 {kecamatan}, {kota}</span><span>{luas_tanah}/{luas_bangunan} m²</span>'
                f'<span>{kamar_tidur} KT · {kamar_mandi} KM</span><span>{lantai} lantai</span></div></div>'
            )

            # 3) Cek kewajaran (jika harga penawaran diisi)
            if penawaran_juta > 0:
                penawaran = penawaran_juta * 1e6
                selisih = (penawaran - estimasi) / estimasi * 100   # + berarti lebih mahal dari estimasi
                if abs(selisih) <= info["mape_test_persen"]:
                    kelas, status, saran = "wajar", "WAJAR", "Harga masih dalam kisaran estimasi pasar."
                elif selisih > 0:
                    kelas, status, saran = "mahal", "DI ATAS PASAR", "Ada ruang negosiasi, kecuali rumah punya keunggulan yang tidak tercatat di data."
                else:
                    kelas, status, saran = "murah", "DI BAWAH PASAR", "Bisa jadi kesempatan bagus; periksa dulu legalitas dan kondisi fisik rumah."

                # Skala visual: posisi kisaran wajar, estimasi, dan penawaran
                kiri_skala = min(batas_bawah, penawaran) * 0.85
                kanan_skala = max(batas_atas, penawaran) * 1.10

                def posisi(x):
                    return (x - kiri_skala) / (kanan_skala - kiri_skala) * 100

                skala = (
                    '<div class="skala">'
                    f'<div class="skala-wajar" style="left:{posisi(batas_bawah):.1f}%;width:{posisi(batas_atas) - posisi(batas_bawah):.1f}%"></div>'
                    f'<div class="penanda" style="left:{posisi(estimasi):.1f}%;background:{WARNA_UTAMA_GELAP}"></div>'
                    f'<div class="penanda-teks" style="left:{posisi(estimasi):.1f}%;top:-24px;color:{WARNA_UTAMA_GELAP}">Estimasi</div>'
                    f'<div class="penanda" style="left:{posisi(penawaran):.1f}%;background:{WARNA_AKSEN}"></div>'
                    f'<div class="penanda-teks" style="left:{posisi(penawaran):.1f}%;top:22px;color:#9A6A00">Penawaran</div>'
                    '</div>'
                )
                html(
                    '<div class="kartu" style="margin-top:14px"><div class="label-kecil">Cek kewajaran harga</div>'
                    f'<div class="pil {kelas}">{status}</div>'
                    f'<div class="teks-redup">Penawaran <b>{format_rp(penawaran)}</b> · selisih <b>{"+" if selisih > 0 else ""}{koma(selisih)}%</b> '
                    f'dari estimasi (toleransi ±{koma(info["mape_test_persen"])}%). {saran}</div>'
                    f'{skala}<div class="teks-redup" style="font-size:12px">Area hijau = kisaran wajar.</div></div>'
                )
            else:
                html('<div class="kartu" style="margin-top:14px"><div class="label-kecil">Cek kewajaran harga</div>'
                     '<div class="teks-redup" style="margin-top:8px">Isi <b>harga penawaran</b> untuk melihat apakah '
                     'harga dari penjual wajar, terlalu mahal, atau terlalu murah.</div></div>')

            html(
                '<div class="grid-2">'
                f'<div class="kartu"><div class="label-kecil">Harga per m² bangunan</div>'
                f'<div class="angka-sedang">{format_rp_lengkap(estimasi / luas_bangunan)}</div></div>'
                f'<div class="kartu"><div class="label-kecil">Model</div>'
                f'<div class="angka-sedang">{info["nama_model"]}</div>'
                f'<div class="teks-redup">Bandingkan di tab Perbandingan Model</div></div></div>'
            )

            # 4) Peringatan jika input di luar jangkauan data latih
            if luas_bangunan > 1500 or luas_tanah > 3000:
                st.warning("Ukuran rumah ini jauh di atas sebagian besar data latih, jadi estimasi kurang akurat.")
            if luas_bangunan > luas_tanah * lantai:
                st.warning("Luas bangunan lebih besar dari luas tanah × jumlah lantai. Periksa kembali isian.")
        elif "input_terakhir" in st.session_state:
            html('<div class="kosong"><div style="font-size:40px">🔁</div><b>Spesifikasi berubah?</b>'
                 '<div class="teks-redup">Tekan <b>Hitung estimasi harga</b> lagi untuk memperbarui hasil.</div></div>')

# =====================================================================
# TAB 2 — CARI AREA (hasil notebook 04)
# =====================================================================
KELAS_CSS = {"Premium": "premium", "Di atas rata-rata": "atas", "Rata-rata": "rata", "Terjangkau": "terjangkau"}
PRIORITAS = {"Lebih dekat": 0.7, "Seimbang": 0.5, "Lebih hemat": 0.3}     # bobot jarak pada skor

with tab_area:
    html('<div class="judul-bagian">Cari area strategis dekat kampus & kawasan industri</div>'
         '<div class="sub-bagian">Pilih tujuanmu dan budget. Harga memakai <b>rumah standar 70/90</b> '
         '(LB 70 m², LT 90 m², 2 KT, 1 KM) agar antar-area bisa dibandingkan secara adil.</div>')

    if data_area is None:
        st.info("Data area belum tersedia. Upload harga_per_area.csv, titik_acuan.csv, dan "
                "ringkasan_titik_acuan.csv (hasil notebook 04) ke repo GitHub.")
    else:
        area, titik_acuan, ringkasan = data_area
        with st.container(key="kartu_area"):
            jenis = st.pills("Saya mencari rumah dekat", ["Kampus", "Kawasan industri"],
                             selection_mode="single", default="Kampus", key="jenis_tujuan")
            kategori = "Industri" if jenis == "Kawasan industri" else "Kampus"
            a1, a2, a3, a4 = st.columns([2, 1, 1, 1])
            with a1:
                pilihan_tujuan = titik_acuan.loc[titik_acuan["kategori"] == kategori, "nama"].tolist()
                tujuan = st.selectbox("Tujuan", pilihan_tujuan, key="tujuan")
            with a2:
                budget_juta = st.number_input("Budget (juta Rp)", min_value=100, max_value=20_000,
                                              value=900, step=50, key="budget")
            with a3:
                jarak_maks = st.slider("Jarak maksimal (km)", 2, 25, 10, key="jarak_maks")
            with a4:
                prioritas = st.select_slider("Prioritas", list(PRIORITAS), value="Seimbang", key="prioritas")

        budget = budget_juta * 1e6
        t = titik_acuan[titik_acuan["nama"] == tujuan].iloc[0]

        # 1) Ringkasan harga di sekitar tujuan (radius 5 km)
        r = ringkasan[ringkasan["nama"] == tujuan]
        if not r.empty:
            r = r.iloc[0]
            teks_iklan = format_rp(r["median_harga_iklan"]) if pd.notna(r["median_harga_iklan"]) else "-"
            html(
                '<div class="grid-3" style="margin-top:16px">'
                f'<div class="hasil-utama"><div class="label-kecil">Rumah 70/90 di sekitar {tujuan} · 5 km</div>'
                f'<div class="angka-besar" style="font-size:34px">{format_rp(r["harga_standar_median"])}</div>'
                f'<div class="pil {KELAS_CSS.get(r["kelas_harga"], "rata")}">{r["kelas_harga"]}</div>'
                f'<div class="teks-redup">Indeks harga {int(round(r["indeks_harga"]))} (100 = rata-rata Jabodetabek)</div></div>'
                f'<div class="kartu"><div class="label-kecil">Median harga iklan sekitar</div>'
                f'<div class="angka-sedang">{teks_iklan}</div>'
                f'<div class="teks-redup">Semua ukuran rumah, dari {int(r["jumlah_iklan"])} iklan</div></div>'
                f'<div class="kartu"><div class="label-kecil">Rentang antar-area</div>'
                f'<div class="angka-sedang" style="font-size:19px">{format_rp(r["harga_standar_min"])} – '
                f'{format_rp(r["harga_standar_max"])}</div>'
                f'<div class="teks-redup">Rumah 70/90 di {int(r["jumlah_area"])} area dalam radius 5 km</div></div>'
                '</div>'
            )

        # 2) Rekomendasi: area dalam radius, harga <= budget, data cukup -> urutkan dengan skor strategis
        hasil = area.copy()
        hasil["jarak_km"] = jarak_km(hasil["lat"].values, hasil["long"].values, t["lat"], t["long"])
        dalam_radius = hasil[hasil["jarak_km"] <= jarak_maks]
        cocok = dalam_radius[(dalam_radius["harga_standar"] <= budget) & dalam_radius["data_cukup"]].copy()
        bobot = PRIORITAS[prioritas]
        cocok["skor"] = 100 * (bobot * (1 - cocok["jarak_km"] / jarak_maks)
                               + (1 - bobot) * (1 - cocok["harga_standar"] / budget))
        cocok = cocok.sort_values("skor", ascending=False).head(5)

        kolom_daftar, kolom_peta = st.columns([1.25, 1], gap="large")
        with kolom_daftar:
            html(f'<div class="judul-bagian" style="font-size:18px">{len(cocok)} area rekomendasi</div>'
                 f'<div class="sub-bagian" style="margin-bottom:0">Skor strategis (0–100) menggabungkan jarak ke '
                 f'{tujuan} dan sisa budget. Hanya area dengan ≥ 5 iklan.</div>')
            if cocok.empty:
                termurah = dalam_radius.loc[dalam_radius["data_cukup"], "harga_standar"]
                saran = (f" Rumah 70/90 termurah dalam radius {jarak_maks} km sekitar {format_rp(termurah.min())}."
                         if not termurah.empty else " Coba perbesar jarak maksimal.")
                st.info("Belum ada area yang cocok dengan budget dan jarak ini." + saran)
            else:
                kartu = ""
                for i, b in enumerate(cocok.itertuples(), start=1):
                    kartu += (
                        f'<div class="kartu-rek{" rek-pertama" if i == 1 else ""}">'
                        f'<div class="nomor">{i}</div>'
                        f'<div class="rek-nama"><div class="nama-area">{b.area}</div>'
                        f'<div class="teks-redup">{b.city} · {b.jumlah_iklan} iklan</div></div>'
                        f'<div class="rek-jarak"><div class="teks-redup">Jarak</div><div class="nilai">{koma(b.jarak_km)} km</div></div>'
                        f'<div><div class="teks-redup">Rumah 70/90</div><div class="nilai">{format_rp(b.harga_standar)}</div>'
                        f'<div class="teks-redup">sisa {format_rp(budget - b.harga_standar)}</div></div>'
                        f'<div class="rek-skor"><div class="teks-redup">Skor strategis <b style="color:{WARNA_TEKS}">{b.skor:.0f}</b></div>'
                        f'<div class="batang"><div class="isi-batang" style="width:{max(b.skor, 3):.0f}%"></div></div></div>'
                        '</div>'
                    )
                html(kartu)
        with kolom_peta:
            # Peta: merah = tujuan, hijau tua = rekomendasi, abu = area lain dalam radius
            peta = pd.concat([
                pd.DataFrame({"lat": dalam_radius["lat"], "lon": dalam_radius["long"], "warna": "#A7B4B1", "ukuran": 110}),
                pd.DataFrame({"lat": cocok["lat"], "lon": cocok["long"], "warna": WARNA_UTAMA, "ukuran": 260}),
                pd.DataFrame({"lat": [t["lat"]], "lon": [t["long"]], "warna": ["#D7301F"], "ukuran": [420]}),
            ], ignore_index=True)
            html('<div style="height:18px"></div>')
            st.map(peta, latitude="lat", longitude="lon", color="warna", size="ukuran", zoom=11, height=420)
            st.caption(f"🔴 {tujuan} · 🟢 area rekomendasi · ⚪ area lain dalam radius {jarak_maks} km. "
                       "Jarak = garis lurus dari titik tengah area (perkiraan).")

# =====================================================================
# TAB 3 — PERBANDINGAN MODEL
# =====================================================================
with tab_model:
    nama_utama = info["nama_model"]
    html(f'<div class="judul-bagian">Perbandingan model</div>'
         f'<div class="sub-bagian">Semua model dilatih dan diuji pada data yang sama (631 rumah data uji). '
         f'Versi aplikasi ini memakai <b>{nama_utama}</b>. MAPE makin kecil makin akurat.</div>')

    if perbandingan:
        mape_maks = max(v.get("mape_test_persen", 0) for v in perbandingan.values()) or 1
        kartu = ""
        for nama, m in perbandingan.items():
            dipakai = nama == nama_utama
            mape_m = m.get("mape_test_persen")
            baris = ""
            if "mae_test_rp" in m:
                baris += f'<div class="baris-metrik"><span>MAE</span><b>{format_rp(m["mae_test_rp"])}</b></div>'
            if "r2_log_test" in m:
                baris += f'<div class="baris-metrik"><span>R² uji (log)</span><b>{koma(m["r2_log_test"], 3)}</b></div>'
            if "r2_log_train" in m and pd.notna(m["r2_log_train"]):
                baris += f'<div class="baris-metrik"><span>R² latih (log)</span><b>{koma(m["r2_log_train"], 3)}</b></div>'
            kartu += (
                f'<div class="kartu{" kartu-dipakai" if dipakai else ""}">'
                f'<div class="nama-model">{nama}{"<span class=lencana-kecil>DIPAKAI</span>" if dipakai else ""}</div>'
                f'<div class="label-kecil" style="margin-top:12px">MAPE data uji</div>'
                f'<div class="angka-sedang">{koma(mape_m, 2) if mape_m else "-"}%</div>'
                f'<div class="batang"><div class="isi-batang" style="width:{(mape_m or 0) / mape_maks * 100:.0f}%"></div></div>'
                f'{baris}</div>'
            )
        html(f'<div class="grid-model">{kartu}</div>')

    # Estimasi semua model untuk rumah terakhir yang dihitung di tab Cek Harga
    html('<div class="judul-bagian" style="font-size:18px;margin-top:28px">Estimasi untuk rumah yang sama</div>')
    if "input_terakhir" not in st.session_state:
        st.info("Hitung dulu sebuah rumah di tab **Cek Harga**, lalu kembali ke sini untuk melihat estimasi dari setiap model.")
    else:
        data_input = st.session_state["input_terakhir"]
        semua_model = {n: (model if n == nama_utama else model_pembanding[n])
                       for n in FILE_SEMUA_MODEL if n == nama_utama or n in model_pembanding}
        est_utama = float(np.exp(model.predict(data_input)[0]))
        kartu = ""
        for nama, m in semua_model.items():
            est = float(np.exp(m.predict(data_input)[0]))
            beda = (est - est_utama) / est_utama * 100
            dipakai = nama == nama_utama
            keterangan = "Model utama versi ini" if dipakai else f"{'+' if beda >= 0 else ''}{koma(beda)}% dari {nama_utama}"
            kartu += (
                f'<div class="kartu{" kartu-dipakai" if dipakai else ""}">'
                f'<div class="nama-model">{nama}{"<span class=lencana-kecil>DIPAKAI</span>" if dipakai else ""}</div>'
                f'<div class="angka-sedang" style="margin-top:10px">{format_rp(est)}</div>'
                f'<div class="teks-redup">{keterangan}</div></div>'
            )
        html(f'<div class="sub-bagian">{st.session_state.get("ringkasan_input", "")}</div>'
             f'<div class="grid-model">{kartu}</div>')
        if len(semua_model) == 1:
            st.caption("File model lain belum di-upload ke repo, jadi hanya model utama yang ditampilkan.")

# =====================================================================
# TAB 4 — TENTANG
# =====================================================================
with tab_tentang:
    html(
        '<div class="judul-bagian">Bagaimana aplikasi ini bekerja</div>'
        '<div class="sub-bagian">Alur dari data mentah sampai estimasi yang kamu lihat.</div>'
        '<div class="alur">'
        '<div><i>01</i><b>Data</b>3.553 iklan rumah Jabodetabek (Kaggle, scraping rumah123.com).</div>'
        '<div><i>02</i><b>Pembersihan</b>Hapus duplikat & nilai tak wajar → 3.153 rumah. Kecamatan diubah jadi koordinat.</div>'
        '<div><i>03</i><b>Preprocessing</b>Imputasi, log-transform, scaling, one-hot encoding (14 → 33 kolom).</div>'
        '<div><i>04</i><b>Modeling</b>Beberapa algoritma dilatih pada log(harga), dituning dengan Grid Search + 5-Fold CV.</div>'
        f'<div><i>05</i><b>Evaluasi</b>Diuji pada 631 rumah. Model versi ini: MAPE ±{koma(info["mape_test_persen"])}%.</div>'
        '</div>'
    )
    t1, t2 = st.columns(2, gap="large")
    with t1:
        html(
            '<div class="judul-bagian" style="font-size:18px">Aturan cek kewajaran</div>'
            '<div class="kartu"><div class="teks-redup">'
            f'Selisih harga penawaran terhadap estimasi dibandingkan dengan MAPE model (±{koma(info["mape_test_persen"])}%):<br><br>'
            '<span class="pil wajar">WAJAR</span> selisih masih di dalam toleransi<br>'
            '<span class="pil mahal">DI ATAS PASAR</span> lebih mahal dari toleransi<br>'
            '<span class="pil murah">DI BAWAH PASAR</span> lebih murah dari toleransi'
            '</div></div>'
        )
    with t2:
        html(
            '<div class="judul-bagian" style="font-size:18px">Keterbatasan</div>'
            '<div class="kartu"><div class="teks-redup">'
            '• Harga di data adalah <b>harga iklan</b>, bukan harga transaksi akhir.<br>'
            '• Lokasi dihitung dari <b>titik tengah kecamatan/area</b>, bukan alamat persis.<br>'
            '• Data Jakarta Pusat dan rumah &gt; Rp10 miliar sedikit, sehingga estimasinya kurang akurat.<br>'
            '• Kondisi yang tidak tercatat (hadap jalan, banjir, akses tol) tidak ikut dihitung.<br>'
            '• Estimasi ini <b>bukan penilaian (appraisal) resmi</b>.'
            '</div></div>'
        )

html(f'<div class="footer"><b>{NAMA_BRAND}</b> · Proyek Akhir Pembelajaran Mesin · {LABEL_VERSI}<br>{NAMA_TIM}</div>')
