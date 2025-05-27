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

# Fungsi utama untuk memproses gambar dan mengekstrak warna dominan dengan K-Means
def process_image(image, min_k=1, max_k=5, threshold=0.005):
    # Mengubah gambar menjadi rgb
    image_rgb = image.convert("RGB")
    img_array = np.array(image_rgb)
    pixels = img_array.reshape(-1, 3)
    
    # OPTIMASI: Convert ke float32 untuk kecepatan dan efisiensi memory
    pixels = pixels.astype(np.float32)
    
    # OPTIMASI: Pilih algoritma berdasarkan ukuran data untuk kecepatan optimal
    use_minibatch = len(pixels) > 100000
    
    # Mencari nilai k optimal
    best_k = max_k
    for k in range(max_k, min_k - 1, -1):
        if use_minibatch:
            # MiniBatchKMeans untuk gambar besar - 5-10x lebih cepat
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                random_state=42, 
                batch_size=min(1000, len(pixels)//10),
                max_iter=100,
                n_init=3
            ).fit(pixels)
        else:
            # KMeans dengan optimasi untuk gambar kecil-menengah
            kmeans = KMeans(
                n_clusters=k, 
                random_state=42, 
                n_init=5,
                max_iter=100,
                algorithm='elkan',
                n_jobs=-1
            ).fit(pixels)
            
        counts = Counter(kmeans.labels_)
        proportions = np.array(list(counts.values())) / len(pixels)
        if all(p >= threshold for p in proportions):
            best_k = k
            break
    
    # Clustering final dengan jumlah k terbaik
    if use_minibatch:
        kmeans = MiniBatchKMeans(
            n_clusters=best_k, 
            random_state=42, 
            batch_size=min(1000, len(pixels)//10),
            max_iter=150,
            n_init=5
        ).fit(pixels)
    else:
        kmeans = KMeans(
            n_clusters=best_k, 
            random_state=42, 
            n_init=10,
            max_iter=300,
            algorithm='elkan',
            n_jobs=-1
        ).fit(pixels)
    
    counts = Counter(kmeans.labels_)

    sorted_indices = np.argsort([count for label, count in counts.most_common()])[::-1]

    top_colors = kmeans.cluster_centers_[sorted_indices]
    hex_colors = [rgb_to_hex(color) for color in top_colors]

    label_mapping = {original_idx: dominant_idx 
                    for dominant_idx, original_idx in enumerate(sorted_indices)}
    
    mapped_labels = np.array([label_mapping[label] for label in kmeans.labels_])
    
    return {
        'top_colors': top_colors,
        'hex_colors': hex_colors,
        'pixels': pixels,
        'labels': mapped_labels,
        'original_centers': kmeans.cluster_centers_,
        'size': image.size,
        'total_pixels': len(pixels),
        'label_mapping': label_mapping,
        'sorted_indices': sorted_indices
    }
    
# Fungsi untuk menandai area warna dominan di dalam gambar
def highlight_dominant_colors(image, dominant_colors):
    img_array = np.array(image.convert("RGB"))
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
        sample_size = min(2000, len(result['pixels']))
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
        with st.spinner("Menganalisis detail pada gambar..."):
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
        with st.spinner("Sedang membuat grafik 3D K-Means Cluster..."):
            plot_3d_colors(result)
    else:
        st.info("Upload gambar untuk memulai!")

# Menjalankan aplikasi
main()
