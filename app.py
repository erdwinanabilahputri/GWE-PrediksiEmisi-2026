# ============================================
# STREAMLIT APP - GWE 2026
# Forecasting Emisi Karbon Perusahaan Energi Global
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="CarbonCast - Prediksi Emisi Karbon",
    page_icon="🌍",
    layout="wide"
)

# ============================================
# FOTO DI SIDEBAR
# ============================================
try:
    st.sidebar.image('foto profil.jpeg', width=150)
except:
    st.sidebar.warning("⚠️ Foto tidak ditemukan")

st.sidebar.markdown("### ERDWINA NABILAH PUTRI")
st.sidebar.caption("GWE 2026 Participant")
st.sidebar.markdown("---")

# ============================================
# TITLE DI SIDEBAR
# ============================================
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA Dashboard", "🔮 Prediction", "📚 Documentation"]
)
st.sidebar.markdown("---")
st.sidebar.caption("GWE 2026 Data Science Challenge")

# ============================================
# LOAD MODEL & DATA - VERSION ANTI ERROR
# ============================================
@st.cache_resource
def load_artifacts():
    """Load model dengan fallback method jika terjadi error"""
    try:
        # Cek folder models
        if not os.path.exists('models'):
            st.error("❌ Folder 'models' tidak ditemukan!")
            return None, None, None
        
        # Load model - coba joblib dulu, kalo error pake pickle
        model = None
        try:
            model = joblib.load('models/model_pipeline.pkl')
        except Exception as e:
            st.warning(f"Joblib error: {e}")
            st.info("Mencoba load dengan pickle...")
            try:
                with open('models/model_pipeline.pkl', 'rb') as f:
                    model = pickle.load(f)
            except Exception as e2:
                st.error(f"❌ Gagal load model: {e2}")
                return None, None, None
        
        # Load preprocessor
        preprocessor = None
        try:
            preprocessor = joblib.load('models/preprocessor.pkl')
        except:
            try:
                with open('models/preprocessor.pkl', 'rb') as f:
                    preprocessor = pickle.load(f)
            except:
                preprocessor = None
        
        # Load feature names
        feature_names = None
        try:
            with open('models/feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        except:
            feature_names = None
        
        st.success("✅ Model berhasil dimuat!")
        return model, preprocessor, feature_names
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load dataset dari file CSV"""
    try:
        if not os.path.exists('emissions_high_granularity.csv'):
            st.error("❌ File 'emissions_high_granularity.csv' tidak ditemukan!")
            return None
        df = pd.read_csv('emissions_high_granularity.csv')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Load semua artifacts
model, preprocessor, feature_names = load_artifacts()
df = load_data()

# ============================================
# PAGE 1: HOME
# ============================================
if page == "🏠 Home":
    st.title("🌍 CarbonCast")
    st.subheader("Smart Industrial Carbon Emission Forecasting")
    
    st.markdown("""
    ---
    ### 📋 Tentang Aplikasi
    Aplikasi ini digunakan untuk **memprediksi emisi karbon** perusahaan energi 
    berdasarkan data historis dari tahun **1962-2022**.
    
    ### 🎯 Tujuan
    - Menganalisis tren emisi karbon global
    - Memprediksi emisi masa depan
    - Mendukung perencanaan kebijakan **Smart City**
    
    ### 📊 Fitur Aplikasi
    1. **📊 EDA Dashboard** - Visualisasi interaktif data emisi
    2. **🔮 Prediction** - Prediksi emisi berdasarkan parameter
    3. **📚 Documentation** - Informasi model dan tim
    ---
    """)
    
    # Statistik ringkas
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Data", f"{len(df):,}")
        with col2:
            st.metric("📅 Periode", f"{df['year'].min()} - {df['year'].max()}")
        with col3:
            st.metric("🏢 Perusahaan", f"{df['parent_entity'].nunique()}")
        with col4:
            st.metric("📈 Rata-rata Emisi", f"{df['total_emissions_MtCO2e'].mean():.2f} Mt")

# ============================================
# PAGE 2: EDA DASHBOARD
# ============================================
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    st.subheader("Eksplorasi Data Emisi Karbon Interaktif")
    
    if df is None:
        st.warning("⚠️ Data tidak tersedia. Silakan periksa file CSV.")
    else:
        # Filter
        st.markdown("### 🔍 Filter Data")
        col1, col2 = st.columns(2)
        with col1:
            companies = ['All'] + sorted(df['parent_entity'].unique().tolist())
            selected_company = st.selectbox("Pilih Perusahaan", companies)
        with col2:
            commodities = ['All'] + sorted(df['commodity'].unique().tolist())
            selected_commodity = st.selectbox("Pilih Komoditas", commodities)
        
        # Filter data
        df_filtered = df.copy()
        if selected_company != 'All':
            df_filtered = df_filtered[df_filtered['parent_entity'] == selected_company]
        if selected_commodity != 'All':
            df_filtered = df_filtered[df_filtered['commodity'] == selected_commodity]
        
        st.markdown(f"**Menampilkan {len(df_filtered):,} baris data**")
        st.markdown("---")
        
        # Visualisasi 1: Time Series
        st.subheader("📈 Tren Emisi per Tahun")
        yearly_data = df_filtered.groupby('year')['total_emissions_MtCO2e'].sum().reset_index()
        fig1 = px.line(
            yearly_data,
            x='year', 
            y='total_emissions_MtCO2e',
            title='Total Emisi per Tahun',
            labels={'year': 'Tahun', 'total_emissions_MtCO2e': 'Total Emisi (Mt CO2e)'},
            markers=True
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Visualisasi 2: Top Perusahaan
        st.subheader("🏢 Top 10 Perusahaan dengan Emisi Tertinggi")
        top_companies = df_filtered.groupby('parent_entity')['total_emissions_MtCO2e'].sum().nlargest(10).reset_index()
        fig2 = px.bar(
            top_companies,
            x='parent_entity', 
            y='total_emissions_MtCO2e',
            title='Top 10 Perusahaan',
            labels={'parent_entity': 'Perusahaan', 'total_emissions_MtCO2e': 'Total Emisi (Mt CO2e)'},
            color='total_emissions_MtCO2e',
            color_continuous_scale='Blues'
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Visualisasi 3: Emisi per Komoditas
        st.subheader("🛢️ Distribusi Emisi per Komoditas")
        commodity_emissions = df_filtered.groupby('commodity')['total_emissions_MtCO2e'].sum().reset_index()
        fig3 = px.pie(
            commodity_emissions,
            values='total_emissions_MtCO2e', 
            names='commodity',
            title='Distribusi Emisi per Komoditas',
            hole=0.3
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

# ============================================
# PAGE 3: PREDICTION
# ============================================
elif page == "🔮 Prediction":
    st.title("🔮 Prediction")
    st.subheader("Prediksi Emisi Karbon")
    
    if model is None:
        st.warning("⚠️ Model tidak tersedia. Silakan periksa file model di folder 'models/'.")
        with st.expander("📁 Cek file yang ada:"):
            if os.path.exists('models'):
                for file in os.listdir('models'):
                    st.write(f"   - {file}")
            else:
                st.write("❌ Folder 'models' tidak ditemukan!")
    elif df is None:
        st.warning("⚠️ Data tidak tersedia. Silakan periksa file CSV.")
    else:
        st.markdown("""
        ### Masukkan Parameter Prediksi
        Isi semua parameter di bawah ini untuk mendapatkan prediksi emisi.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.slider("📅 Tahun", 2023, 2030, 2025, help="Tahun yang ingin diprediksi")
            company = st.selectbox("🏢 Perusahaan", df['parent_entity'].unique())
            commodity = st.selectbox("🛢️ Komoditas", df['commodity'].unique())
        
        with col2:
            production_value = st.number_input("📊 Nilai Produksi", min_value=0.0, value=500.0, step=10.0)
            parent_type = st.selectbox("🏛️ Tipe Perusahaan", df['parent_type'].unique())
            reporting_entity = st.selectbox("📋 Entitas Pelapor", df['reporting_entity'].unique())
            production_unit = st.selectbox("📏 Unit Produksi", df['production_unit'].unique())
            source = st.text_input("📝 Sumber Data", "User Input")
        
        st.markdown("---")
        
        if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
            # Buat input dataframe
            input_data = pd.DataFrame({
                'year': [year],
                'parent_entity': [company],
                'parent_type': [parent_type],
                'reporting_entity': [reporting_entity],
                'commodity': [commodity],
                'production_value': [production_value],
                'production_unit': [production_unit],
                'source': [source]
            })
            
            with st.spinner("⏳ Sedang memproses prediksi..."):
                try:
                    prediction = model.predict(input_data)[0]
                    
                    st.success(f"✅ **Prediksi Total Emisi: {prediction:.2f} Mt CO2e**")
                    
                    # Interpretasi
                    st.subheader("📊 Interpretasi")
                    if prediction > 500:
                        st.warning("⚠️ **Emisi TINGGI** - Perlu upaya reduksi signifikan!")
                        st.progress(0.9)
                    elif prediction > 200:
                        st.info("ℹ️ **Emisi SEDANG** - Masih perlu perbaikan.")
                        st.progress(0.6)
                    else:
                        st.success("✅ **Emisi RENDAH** - Sudah baik, pertahankan!")
                        st.progress(0.3)
                    
                    # Tampilkan parameter yang digunakan
                    with st.expander("📋 Lihat Parameter yang Digunakan"):
                        st.json({
                            'year': year,
                            'company': company,
                            'commodity': commodity,
                            'production_value': production_value,
                            'parent_type': parent_type,
                            'reporting_entity': reporting_entity,
                            'production_unit': production_unit,
                            'source': source
                        })
                    
                except Exception as e:
                    st.error(f"❌ Error saat prediksi: {e}")

# ============================================
# PAGE 4: DOCUMENTATION
# ============================================
else:
    st.title("📚 Documentation")
    
    # FOTO DI HALAMAN DOCUMENTATION
    st.markdown("### 👤 Pengembang")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image('foto profil.jpeg', width=250)
        except:
            st.warning("⚠️ Foto tidak ditemukan")
        st.markdown("### Erdwina Nabilah Putri")
        st.caption("GWE 2026 Participant")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🤖 Model yang Digunakan
    
    **Model Terbaik:** Gradient Boosting Regressor
    
    | Metrik | Nilai |
    |--------|-------|
    | R2 Score | **0.9926** |
    | MAE | 11.03 Mt CO2e |
    | RMSE | 23.68 Mt CO2e |
    | CV R2 | 0.9900 |
    
    ---
    
    ### 📊 Fitur Paling Penting
    Berdasarkan analisis feature importance:
    
    1. **Production Value** - Nilai produksi (paling dominan)
    2. **Commodity** - Jenis komoditas
    3. **Year** - Tahun
    4. **Parent Entity** - Perusahaan
    
    ---
    
    ### 📁 Sumber Data
    - **Sumber:** Climate Accountability Institute (CAI)
    - **Periode:** 1962-2022
    - **Jumlah Data:** 15,797 baris
    
    ---
    
    ### 🛠️ Tools yang Digunakan
    - Python, Pandas, NumPy
    - Scikit-learn, XGBoost, Gradient Boosting
    - Streamlit, Plotly
    - SHAP (Interpretability)
    
    ---
    
    ### 📌 GWE 2026
    Proyek ini dibuat untuk **GWE 2026 Data Science Challenge**
    dengan tema **Forecasting + Smart City**.
    """)

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.caption("🌍 GWE 2026 Data Science Challenge")