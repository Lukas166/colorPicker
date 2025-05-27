# Import library yang diperlukan
import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import plotly.graph_objects as go
import io

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Color Palette Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk mengubah warna RGB menjadi HEX
def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

# Fungsi untuk mengubah ukuran gambar agar lebih cepat diproses
def resize_image_for_processing(image, max_size=800):
    """Resize gambar untuk mempercepat proses clustering sambil mempertahankan aspect ratio"""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image

# Fungsi untuk sampling pixel secara acak agar lebih efisien
def sample_pixels(pixels, max_samples=50000):
    # OPTIMASI: Tidak lagi sampling, return semua pixels untuk analisis lengkap
    return pixels.astype(np.float32)  # Convert ke float32 untuk optimasi memory dan speed

# Fungsi untuk mengecek kompleksitas warna dalam gambar
def check_color_complexity(pixels):
    # OPTIMASI: Gunakan sample kecil hanya untuk cek kompleksitas
    sample_size = min(5000, len(pixels))
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[indices]
    else:
        sample_pixels = pixels
        
    std_dev = np.std(sample_pixels, axis=0)
    mean_std = np.mean(std_dev)
    
    # Jika variasi warna sangat kecil, kemungkinan gambar monokrom
    if mean_std < 10:
        return 1  # Gambar hampir monokrom, cukup 1 warna
    elif mean_std < 25:
        return 2  # Gambar sederhana, cukup 2 warna
    elif mean_std < 50:
        return 3  # Gambar moderate, mulai dari 3 warna
    else:
        return None  # Gambar kompleks, gunakan algoritma penuh

# Fungsi utama untuk memproses gambar dan mengekstrak warna dominan dengan K-Means
def process_image(image, min_k=1, max_k=5, threshold=0.0001):
    # OPTIMASI: Tidak resize gambar untuk analisis, proses semua pixel asli
    # processed_image = resize_image_for_processing(image)
    
    # Mengubah gambar menjadi rgb
    image_rgb = image.convert("RGB")
    img_array = np.array(image_rgb)
    pixels = img_array.reshape(-1, 3)
    
    # OPTIMASI: Proses semua pixel dengan optimasi kecepatan
    all_pixels = sample_pixels(pixels)  # Sekarang return semua pixels yang dioptimasi
    
    # Cek kompleksitas warna gambar untuk optimasi
    suggested_k = check_color_complexity(all_pixels)
    
    best_k = max_k
    best_kmeans = None
    
    # OPTIMASI: Pilih algoritma berdasarkan ukuran data
    use_minibatch = len(all_pixels) > 100000  # Gunakan MiniBatch untuk gambar besar
    
    # Jika gambar sederhana, langsung gunakan k yang disarankan
    if suggested_k is not None and suggested_k <= 2:
        if use_minibatch:
            # OPTIMASI: MiniBatchKMeans untuk gambar besar
            kmeans = MiniBatchKMeans(
                n_clusters=suggested_k, 
                random_state=42, 
                batch_size=min(1000, len(all_pixels)//10),
                max_iter=100,
                n_init=3
            ).fit(all_pixels)
        else:
            # KMeans biasa untuk gambar kecil
            kmeans = KMeans(
                n_clusters=suggested_k, 
                random_state=42, 
                n_init=5,
                max_iter=100,
                algorithm='elkan',
                n_jobs=-1
            ).fit(all_pixels)
        best_k = suggested_k
        best_kmeans = kmeans
    else:
        # Untuk gambar kompleks, mulai dari k=3 dan naik (lebih efisien)
        for k in range(3, max_k + 1):
            if use_minibatch:
                # OPTIMASI: MiniBatchKMeans untuk kecepatan tinggi
                kmeans = MiniBatchKMeans(
                    n_clusters=k, 
                    random_state=42,
                    batch_size=min(1000, len(all_pixels)//10),
                    max_iter=100,
                    n_init=3,
                    reassignment_ratio=0.01
                ).fit(all_pixels)
            else:
                # KMeans biasa dengan optimasi
                kmeans = KMeans(
                    n_clusters=k, 
                    random_state=42, 
                    n_init=5,
                    max_iter=100,
                    algorithm='elkan',
                    n_jobs=-1
                ).fit(all_pixels)
                
            counts = Counter(kmeans.labels_)
            proportions = np.array(list(counts.values())) / len(all_pixels)
            if all(p >= threshold for p in proportions):
                best_k = k
                best_kmeans = kmeans
                break
        
        # Jika tidak ada k yang cocok dari 3-5, coba k=2 dan k=1
        if best_kmeans is None:
            for k in [2, 1]:
                if use_minibatch:
                    kmeans = MiniBatchKMeans(
                        n_clusters=k, 
                        random_state=42,
                        batch_size=min(1000, len(all_pixels)//5),
                        max_iter=50,
                        n_init=3
                    ).fit(all_pixels)
                else:
                    kmeans = KMeans(
                        n_clusters=k, 
                        random_state=42, 
                        n_init=5,
                        max_iter=50,
                        n_jobs=-1
                    ).fit(all_pixels)
                    
                counts = Counter(kmeans.labels_)
                proportions = np.array(list(counts.values())) / len(all_pixels)
                if all(p >= threshold for p in proportions):
                    best_k = k
                    best_kmeans = kmeans
                    break
    
    # Gunakan hasil K-Means yang sudah ada (tidak perlu clustering ulang)
    if best_kmeans is None:
        if use_minibatch:
            best_kmeans = MiniBatchKMeans(
                n_clusters=best_k, 
                random_state=42,
                batch_size=min(1000, len(all_pixels)//5),
                max_iter=50,
                n_init=3
            ).fit(all_pixels)
        else:
            best_kmeans = KMeans(
                n_clusters=best_k, 
                random_state=42, 
                n_init=5,
                max_iter=50,
                n_jobs=-1
            ).fit(all_pixels)
    
    counts = Counter(best_kmeans.labels_)
    sorted_indices = np.argsort([count for label, count in counts.most_common()])[::-1]

    top_colors = best_kmeans.cluster_centers_[sorted_indices]
    hex_colors = [rgb_to_hex(color) for color in top_colors]

    label_mapping = {original_idx: dominant_idx 
                    for dominant_idx, original_idx in enumerate(sorted_indices)}
    
    mapped_labels = np.array([label_mapping[label] for label in best_kmeans.labels_])
    
    return {
        'top_colors': top_colors,
        'hex_colors': hex_colors,
        'pixels': all_pixels,  # OPTIMASI: Semua pixels yang sudah diproses
        'labels': mapped_labels,
        'original_centers': best_kmeans.cluster_centers_,
        'size': image.size,  # Ukuran gambar asli
        'total_pixels': len(pixels),  # Total pixel gambar asli
        'label_mapping': label_mapping,
        'sorted_indices': sorted_indices
    }
    
# Fungsi untuk menandai area warna dominan di dalam gambar
def highlight_dominant_colors(image, dominant_colors):
    # Resize gambar untuk mempercepat proses highlight
    processed_image = resize_image_for_processing(image, max_size=600)
    img_array = np.array(processed_image.convert("RGB"))
    output = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for i, color in enumerate(dominant_colors):
        center_bgr = np.array([color[2], color[1], color[0]], dtype=np.uint8)
        lower = np.clip(center_bgr - 20, 0, 255)
        upper = np.clip(center_bgr + 20, 0, 255)
        
        # Membuat mask untuk mendeteksi warna dominan dan menggambar lingkaran
        mask = cv2.inRange(output.copy(), lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest)
            if radius > 5:
                cv2.circle(output, (int(x), int(y)), int(radius), (255, 255, 255), 3)
                cv2.putText(output, str(i+1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Fungsi untuk membuat tampilan palet warna di UI
def create_color_palette(hex_colors):
    st.subheader("Color Palette")
    cols = st.columns(len(hex_colors))
    for i, (col, hex_color) in enumerate(zip(cols, hex_colors)):
        col.markdown(f"""
            <div style='text-align:center; font-weight:bold;'>Color {i+1}</div>
            <div style='background-color:{hex_color}; height:100px; border-radius:10px;'></div>
            <p style='text-align:center;'>{hex_color}</p>
        """, unsafe_allow_html=True)

# Fungsi untuk menghasilkan gambar palet warna yang dapat diunduh
def generate_palette_image(hex_colors):
    width, height = 200 * len(hex_colors), 140
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for i, hex_color in enumerate(hex_colors):
        x0 = i * 200
        draw.rectangle([x0, 0, x0 + 200, 100], fill=hex_color)
        text = hex_color
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x0 + (200 - text_width)/2, 110), text, fill="black", font=font)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Fungsi untuk membuat plot 3D warna dalam ruang RGB
def plot_3d_colors(result):
    fig = go.Figure()
    
    # Plot titik-titik pixel
    if 'pixels' in result and 'labels' in result:
        # OPTIMASI: Sample untuk visualisasi 3D agar tidak lag, tapi analisis tetap full
        sample_size = min(3000, len(result['pixels']))
        indices = np.random.choice(len(result['pixels']), sample_size, replace=False)
        sampled_pixels = result['pixels'][indices]
        sampled_labels = result['labels'][indices]
        
        for cluster_idx in range(len(result['top_colors'])):
            cluster_points = sampled_pixels[sampled_labels == cluster_idx]
            if len(cluster_points) > 0:
                fig.add_trace(go.Scatter3d(
                    x=cluster_points[:,0],
                    y=cluster_points[:,1],
                    z=cluster_points[:,2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=result['hex_colors'][cluster_idx],  # Warna sesuai urutan
                        opacity=0.6
                    ),
                    name=f'Cluster {cluster_idx+1}'
                ))
    
    # Plot centroid
    for idx, color in enumerate(result['hex_colors']):
        fig.add_trace(go.Scatter3d(
            x=[result['top_colors'][idx,0]],
            y=[result['top_colors'][idx,1]],
            z=[result['top_colors'][idx,2]],
            mode='markers+text',
            marker=dict(
                size=10,
                color=color,
                symbol='diamond',
                line=dict(width=1, color='white')
            ),
            text=f'Color {idx+1}',
            textposition='top center',
            name=f'Centroid {idx+1}',
            showlegend=False
        ))
    
    # Konfigurasi layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            bgcolor='rgba(0,0,0,0.9)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fungsi utama untuk menjalankan aplikasi
def main():
    st.title("Image Color Palette Generator")
    st.markdown("Upload gambar untuk mendapatkan color palette menggunakan K-Means clustering.")
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Proses analisis gambar dan ekstraksi warna dominan
        with st.spinner("Menganalisis detail gambar..."):
            result = process_image(image)
            
        # Menampilkan gambar asli dan hasil deteksi warna
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(highlight_dominant_colors(image, result['top_colors']), 
                     caption="Detected Dominant Colors", use_container_width=True)
        
        # Informasi gambar dan jumlah warna
        st.markdown(f"""
        **Resolusi gambar:** {result['size'][0]} x {result['size'][1]} pixels  
        **Total pixel yang dianalisis:** {result['total_pixels']:,}  
        **Warna dominan terdeteksi (1-5):** {len(result['top_colors'])}
        """)
        
        # Tampilkan palet warna dominan
        create_color_palette(result['hex_colors'])
        
        # Tombol untuk mengunduh palet sebagai gambar
        palette_img = generate_palette_image(result['hex_colors'])
        st.download_button(
            "Download Color Palette sebagai Gambar (.png)",
            data=palette_img,
            file_name="palette.png",
            mime="image/png"
        )
        
        # Tampilkan grafik 3D dari distribusi warna dengan loading terpisah
        with st.spinner("Tunggu sebentar untuk menghasilkan grafik 3D K-Means Cluster..."):
            plot_3d_colors(result)
    else:
        st.info("Upload gambar untuk memulai!")

# Menjalankan aplikasi
main()
