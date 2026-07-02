# ============================================
# STREAMLIT APP - VERSION DENGAN PREPROCESSOR
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
# SIDEBAR
# ============================================
try:
    st.sidebar.image('foto profil.jpeg', width=150)
except:
    st.sidebar.warning("⚠️ Foto tidak ditemukan")

st.sidebar.markdown("### ERDWINA NABILAH PUTRI")
st.sidebar.caption("GWE 2026 Participant")
st.sidebar.markdown("---")

st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA Dashboard", "🔮 Prediction", "📚 Documentation"]
)
st.sidebar.markdown("---")
st.sidebar.caption("🌍 GWE 2026")

# ============================================
# LOAD MODEL & PREPROCESSOR
# ============================================
@st.cache_resource
def load_artifacts():
    """Load model dan preprocessor"""
    try:
        if not os.path.exists('models'):
            st.error("❌ Folder 'models' tidak ditemukan!")
            return None, None
        
        # Load model
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessor
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        st.success("✅ Model dan preprocessor berhasil dimuat!")
        return model, preprocessor
        
    except Exception as e:
        st.error(f"❌ Error loading: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('emissions_high_granularity.csv')
        return df
    except:
        st.error("❌ File CSV tidak ditemukan!")
        return None

# Load semua artifacts
model, preprocessor = load_artifacts()
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
    Aplikasi ini digunakan untuk **memprediksi emisi karbon** perusahaan energi.
    
    ### 🎯 Tujuan
    - Menganalisis tren emisi karbon global
    - Memprediksi emisi masa depan
    - Mendukung perencanaan kebijakan **Smart City**
    
    ### 📊 Fitur Aplikasi
    1. **📊 EDA Dashboard** - Visualisasi data emisi
    2. **🔮 Prediction** - Prediksi emisi
    3. **📚 Documentation** - Informasi model
    ---
    """)
    
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
    
    if df is None:
        st.warning("⚠️ Data tidak tersedia.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            companies = ['All'] + sorted(df['parent_entity'].unique().tolist())
            selected_company = st.selectbox("Pilih Perusahaan", companies)
        with col2:
            commodities = ['All'] + sorted(df['commodity'].unique().tolist())
            selected_commodity = st.selectbox("Pilih Komoditas", commodities)
        
        df_filtered = df.copy()
        if selected_company != 'All':
            df_filtered = df_filtered[df_filtered['parent_entity'] == selected_company]
        if selected_commodity != 'All':
            df_filtered = df_filtered[df_filtered['commodity'] == selected_commodity]
        
        st.markdown(f"**Menampilkan {len(df_filtered):,} baris data**")
        st.markdown("---")
        
        # Tren Emisi
        st.subheader("📈 Tren Emisi per Tahun")
        yearly_data = df_filtered.groupby('year')['total_emissions_MtCO2e'].sum().reset_index()
        fig1 = px.line(yearly_data, x='year', y='total_emissions_MtCO2e', markers=True)
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Top Perusahaan
        st.subheader("🏢 Top 10 Perusahaan")
        top = df_filtered.groupby('parent_entity')['total_emissions_MtCO2e'].sum().nlargest(10).reset_index()
        fig2 = px.bar(top, x='parent_entity', y='total_emissions_MtCO2e', color='total_emissions_MtCO2e')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Per Komoditas
        st.subheader("🛢️ Distribusi per Komoditas")
        komoditas = df_filtered.groupby('commodity')['total_emissions_MtCO2e'].sum().reset_index()
        fig3 = px.pie(komoditas, values='total_emissions_MtCO2e', names='commodity', hole=0.3)
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

# ============================================
# PAGE 3: PREDICTION
# ============================================
elif page == "🔮 Prediction":
    st.title("🔮 Prediction")
    
    if model is None or preprocessor is None:
        st.warning("⚠️ Model atau preprocessor tidak tersedia.")
    elif df is None:
        st.warning("⚠️ Data tidak tersedia.")
    else:
        st.markdown("### Masukkan Parameter Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.slider("📅 Tahun", 2023, 2030, 2025)
            company = st.selectbox("🏢 Perusahaan", df['parent_entity'].unique())
            commodity = st.selectbox("🛢️ Komoditas", df['commodity'].unique())
        
        with col2:
            production_value = st.number_input("📊 Nilai Produksi", min_value=0.0, value=500.0, step=10.0)
            parent_type = st.selectbox("🏛️ Tipe Perusahaan", df['parent_type'].unique())
            reporting_entity = st.selectbox("📋 Entitas Pelapor", df['reporting_entity'].unique())
            production_unit = st.selectbox("📏 Unit Produksi", df['production_unit'].unique())
            source = st.text_input("📝 Sumber Data", "User Input")
        
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
            
            with st.spinner("⏳ Memproses..."):
                try:
                    # Proses input dengan preprocessor
                    input_processed = preprocessor.transform(input_data)
                    
                    # Prediksi
                    prediction = model.predict(input_processed)[0]
                    
                    st.success(f"✅ **Prediksi Emisi: {prediction:.2f} Mt CO2e**")
                    
                    if prediction > 500:
                        st.warning("⚠️ **Emisi TINGGI**")
                        st.progress(0.9)
                    elif prediction > 200:
                        st.info("ℹ️ **Emisi SEDANG**")
                        st.progress(0.6)
                    else:
                        st.success("✅ **Emisi RENDAH**")
                        st.progress(0.3)
                    
                    with st.expander("📋 Detail Input"):
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
                    st.error(f"❌ Error: {e}")

# ============================================
# PAGE 4: DOCUMENTATION
# ============================================
else:
    st.title("📚 Documentation")
    
    st.markdown("### 👤 Pengembang")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image('foto profil.jpeg', width=200)
        except:
            st.warning("⚠️ Foto tidak ditemukan")
        st.markdown("### Erdwina Nabilah Putri")
        st.caption("GWE 2026 Participant")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🤖 Model
    
    **Model:** Gradient Boosting Regressor
    
    | Metrik | Nilai |
    |--------|-------|
    | R2 Score | **0.9926** |
    | MAE | 11.03 Mt CO2e |
    | RMSE | 23.68 Mt CO2e |
    
    ---
    
    ### 📊 Fitur Penting
    1. **Production Value** - Nilai produksi
    2. **Commodity** - Jenis komoditas
    3. **Year** - Tahun
    4. **Parent Entity** - Perusahaan
    
    ---
    
    ### 📁 Sumber Data
    - **Sumber:** Climate Accountability Institute
    - **Periode:** 1962-2022
    - **Data:** 15,797 baris
    
    ---
    
    ### 🛠️ Tools
    - Python, Pandas, NumPy
    - Scikit-learn, Gradient Boosting
    - Streamlit, Plotly
    
    ---
    
    ### 📌 GWE 2026
    Tema: **Forecasting + Smart City**
    """)

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.caption("🌍 GWE 2026")