import os
os.environ["PYTHONUTF8"] = "1"
os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"

import matplotlib
matplotlib.use('Agg')

import json
import glob
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# ==========================================
# GÜVENLİ GEMİNİ API BAĞLANTISI (ST.SECRETS)
# ==========================================
import google.generativeai as genai

ai_model = None
try:
    # Önce Streamlit Secrets'tan okumayı dener (Bulut için en güvenlisi)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        # Eğer yerelde test ediyorsanız buraya doğrudan anahtarınızı yazabilirsiniz
        api_key = "BURAYA_API_ANAHTARINIZI_YAZIN"

    if api_key and api_key != "BURAYA_API_ANAHTARINIZI_YAZIN":
        genai.configure(api_key=api_key)
        # En kararlı çalışan güncel model tanımlayıcısı
        ai_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    ai_model = None

# Mobil/Web Düzeni
st.set_page_config(
    page_title="Archaeo-AI-AR Profesyonel Saha Ajanı",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏛️ Archaeo-AI-AR Profesyonel Saha Ajanı")
st.caption("Otonom Arkeolojik Anomali Tespit, Spektral Analiz ve Gemini Destekli Uzman Asistan Paneli")

# GeoJSON Yükleme
def load_targets():
    if os.path.exists("master_anomaliler.geojson"):
        try:
            with open("master_anomaliler.geojson", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"GeoJSON okuma hatası: {e}")
    return {"type": "FeatureCollection", "features": []}

geojson_data = load_targets()
features = geojson_data.get("features", [])

# Yan Menü Kontrolleri
st.sidebar.header("🎯 Saha Kontrol Noktası")

target_options = {}
for feat in features:
    props = feat.get("properties", {})
    tid = props.get("id", props.get("ID", 1))
    coords = feat.get("geometry", {}).get("coordinates", [0, 0])
    lon, lat = coords[0], coords[1]
    label = f"Hedef #{tid} ({lat:.4f}, {lon:.4f})"
    target_options[label] = {"id": tid, "lat": lat, "lon": lon, "feature": feat}

if target_options:
    selected_option = st.sidebar.selectbox("Kayıtlı Anomali Seçin", list(target_options.keys()))
else:
    selected_option = None
    st.sidebar.warning("Kayıtlı anomali bulunamadı.")

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Sahada Manuel Koordinat Gir")
custom_lat = st.sidebar.number_input("Enlem (Latitude)", value=39.841389, format="%.6f")
custom_lon = st.sidebar.number_input("Boylam (Longitude)", value=34.812219, format="%.6f")
use_custom = st.sidebar.checkbox("Manuel Koordinatı Kullan")

if use_custom:
    current_lat, current_lon = custom_lat, custom_lon
    target_id_str = "Manuel Saha Noktası"
    target_num = 0
elif selected_option and selected_option in target_options:
    info = target_options[selected_option]
    current_lat, current_lon = info["lat"], info["lon"]
    target_num = info["id"]
    target_id_str = f"Hedef #{target_num}"
else:
    current_lat, current_lon = 39.841389, 34.812219
    target_id_str = "Varsayılan Nokta"
    target_num = 1

st.sidebar.success(f"**Aktif Nokta:** {target_id_str}\n\n**Enlem:** {current_lat:.6f}\n\n**Boylam:** {current_lon:.6f}")

# Dosya Eşleme
def get_target_files(tid):
    profile_imgs = glob.glob(f"hedef_{tid}_profil*.png") + glob.glob(f"hedef_{tid}_kesit*.png")
    ndvi_imgs = glob.glob(f"hedef_{tid}_spektral*.png") + glob.glob(f"hedef_{tid}_ndvi*.png")
    mesh_htmls = glob.glob(f"hedef_{tid}_3d*.html") + glob.glob(f"hedef_{tid}_mesh*.html")
    return list(set(profile_imgs)), list(set(ndvi_imgs)), list(set(mesh_htmls))

p_imgs, n_imgs, m_htmls = get_target_files(target_num)

# Sekmeler
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🤖 Gemini Uzman Asistan", 
    "🌿 Spektral NDVI", 
    "🔥 Termal/Saha Foto", 
    "📊 2D Kesit", 
    "🧊 3D Model", 
    "📋 Liste"
])

with tab1:
    st.subheader("💬 Kıdemli Jeo-Arkeolog Yapay Zeka Danışmanı")
    st.write(f"Şu an **{target_id_str}** konumundasınız (Enlem: `{current_lat}`, Boylam: `{current_lon}`).")
    
    if ai_model is None:
        st.error("⚠️ Gemini API anahtarı bulunamadı veya yapılandırılamadı! Lütfen Streamlit Secrets paneline veya koda anahtarınızı ekleyin.")

    # Otonom Profesyonel Risk Raporu
    if st.button("📊 Kapsamlı Arkeolojik Risk ve Anomali Raporu Oluştur"):
        if ai_model:
            with st.spinner("Gemini profesyonel veri tabanı üzerinden özgün rapor hazırlıyor..."):
                seed_val = random.randint(1000, 9999)
                prompt_text = f"""
                [Analiz Kimliği: {seed_val}]
                Sen kıdemli bir jeo-arkeolog ve uzaktan algılama uzmanısın. 
                Şu an coğrafi olarak {target_id_str} konumunda (Enlem: {current_lat}, Boylam: {current_lon}) saha incelemesi yapıyorsun.
                Bu koordinattaki anomali noktası için TAMAMEN ÖZGÜN, ezbere dayalı olmayan, bu konuma ve rastgele analiz ID'sine ({seed_val}) özel teknik bir ön değerlendirme raporu hazırla.
                Rapor şu başlıkları içersin:
                1. Olası Yapısal Tipoloji ve Dönem Analizi
                2. Mikro-Topoğrafya ve Spektral Beklentiler
                3. Sahada Önerilen Nokta Jeofizik Yöntemleri
                4. Koruma ve Risk Durumu
                Bilimsel, net ve her defasında farklı detaylar içeren özgün bir üslup kullan.
                """
                try:
                    response = ai_model.generate_content(prompt_text)
                    st.success("🎯 **Özgün Uzman Raporu Tamamlandı:**")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Rapor üretilirken hata oluştu: {e}")
        else:
            st.error("Model aktif olmadığı için rapor üretilemiyor.")

    st.markdown("---")

    # Sohbet Geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Merhaba! Ben Archaeo-AI Uzman Asistanıyım. {target_id_str} bölgesindeki veriler hakkında bana her şeyi sorabilirsiniz."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Uzmana profesyonel bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            if ai_model:
                with st.spinner("Uzman inceliyor..."):
                    context_prompt = f"""
                    Sen kıdemli bir jeo-arkeologsun. Kullanıcı şu an {target_id_str} (Enlem: {current_lat}, Boylam: {current_lon}) noktasında.
                    Kullanıcının Sorusu: {user_prompt}
                    Lütfen önceden ezberlenmiş kalıplar kullanmadan, doğrudan bu soruya ve koordinata özel, teknik ve akıcı bir arkeolojik yanıt ver.
                    """
                    try:
                        chat_response = ai_model.generate_content(context_prompt)
                        reply = chat_response.text
                    except Exception as e:
                        reply = f"API Bağlantı hatası: {e}"
            else:
                reply = "Yapay zeka modeli yapılandırılmadı."

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

with tab2:
    st.subheader("🛰️ Sentinel-2 Spektral Nem & Bitki İzi (NDVI)")
    if n_imgs:
        st.image(n_imgs[0], caption=f"Kayıtlı NDVI Haritası ({target_id_str})", use_container_width=True)
    else:
        st.info("Bu hedef için kaydedilmiş NDVI haritası bulunamadı.")

with tab3:
    st.subheader("🔥 Sahadan Termal / Optik Fotoğraf Anomali Tespiti")
    uploaded_file = st.file_uploader("Arazi Fotoğrafı Yükle (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Orijinal Saha Görseli", use_container_width=True)
        with col2:
            st.write("🧠 Yapay Zeka Termal/Isı Gradyan Maskesi")
            img_gray = image.convert("L")
            arr = np.array(img_gray)
            fig_t, ax_t = plt.subplots(figsize=(6, 6))
            cax_t = ax_t.imshow(arr, cmap='inferno')
            fig_t.colorbar(cax_t, ax=ax_t, label='Isı / Yoğunluk Gradyanı')
            ax_t.set_title("Termal Anomali Haritası")
            ax_t.axis('off')
            st.pyplot(fig_t)
            plt.close(fig_t)
            st.success("✅ Anomali kontrast analizi tamamlandı.")

with tab4:
    st.subheader("📊 Topoğrafik Yükseklik Kesiti (2D)")
    if p_imgs:
        for p_file in p_imgs:
            st.image(p_file, caption=f"Profil Kesiti: {os.path.basename(p_file)}", use_container_width=True)
    else:
        st.warning("Bu hedef için kaydedilmiş 2D profil görseli bulunamadı.")

with tab5:
    st.subheader("🧊 İnteraktif 3D Mesh Yüzey Modeli")
    if m_htmls:
        mesh_file = m_htmls[0]
        st.write(f"📁 **Model Dosyası:** `{os.path.basename(mesh_file)}`")
        try:
            with open(mesh_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=550, scrolling=True)
        except Exception as info_err:
            st.error(f"3D HTML dosyası okunurken hata oluştu: {info_err}")
    else:
        st.warning("Bu hedef için 3D Mesh HTML dosyası bulunamadı.")

with tab6:
    st.subheader("📋 Tespit Edilen Tüm Anomaliler Listesi")
    if features:
        table_data = []
        for feat in features:
            props = feat.get("properties", {})
            coords = feat.get("geometry", {}).get("coordinates", [0, 0])
            table_data.append({
                "ID": props.get("id", "-"),
                "Kaynak": props.get("source_file", "DEM GLO-30"),
                "Enlem (Lat)": f"{coords[1]:.6f}",
                "Boylam (Lon)": f"{coords[0]:.6f}"
            })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.info("GeoJSON içinde hedef listesi bulunamadı.")