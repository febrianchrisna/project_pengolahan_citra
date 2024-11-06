import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Fungsi untuk menampilkan histogram
def show_histogram(image):
    gray_image = image.convert('L')
    hist_values = np.histogram(np.array(gray_image).ravel(), bins=256, range=(0, 256))[0]
    plt.figure(figsize=(8, 4))
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.plot(hist_values)
    st.pyplot(plt)

# Fungsi untuk menyimpan gambar
def save_image(image, file_format='JPEG'):
    buffer = BytesIO()
    image.save(buffer, format=file_format)
    st.download_button(
        label="Download Image",
        data=buffer.getvalue(),
        file_name=f'edited_image.{file_format.lower()}',
        mime=f'image/{file_format.lower()}'
    )

st.title("Aplikasi Manipulasi Citra dengan Streamlit")

# Pilih gambar
uploaded_file = st.file_uploader("Upload gambar", type=['jpg', 'jpeg', 'png', 'bmp'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar asli', use_column_width=True)
    st.write("Histogram asli:")
    show_histogram(image)

    # Konversi gambar menjadi format array untuk OpenCV
    img_cv2 = np.array(image.convert('RGB'))
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    # Pilihan fitur manipulasi citra
    st.sidebar.header("Pilih Fitur Manipulasi")
    option = st.sidebar.selectbox("Pilih fitur:", 
                                  ['Grayscale', 'Binary', 'Noise', 'Negative', 'Blur', 'Pseudocolor',
                                   'True Color Filter', 'Edge Detection (Prewitt, Sobel, Canny)'])

    if option == 'Grayscale':
        st.sidebar.write("Tingkat grayscale:")
        if st.sidebar.button("Apply Grayscale"):
            gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            st.image(gray_image, caption='Gambar setelah grayscale', channels='GRAY', use_column_width=True)

    elif option == 'Binary':
        threshold = st.sidebar.slider("Threshold untuk biner", 0, 255, 128)
        if st.sidebar.button("Apply Binary"):
            gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            st.image(binary_image, caption='Gambar setelah biner', channels='GRAY', use_column_width=True)

    elif option == 'Noise':
        noise_level = st.sidebar.slider("Tingkat noise (0-100)", 0, 100, 10)
        if st.sidebar.button("Apply Noise"):
            noise = np.random.randint(0, noise_level, img_cv2.shape, dtype='uint8')
            noisy_image = cv2.add(img_cv2, noise)
            st.image(noisy_image, caption='Gambar dengan noise', channels='BGR', use_column_width=True)

    elif option == 'Negative':
        if st.sidebar.button("Apply Negative"):
            negative_image = cv2.bitwise_not(img_cv2)
            st.image(negative_image, caption='Gambar negatif', channels='BGR', use_column_width=True)

    elif option == 'Blur':
        blur_radius = st.sidebar.slider("Radius blur", 1, 30, 5)
        if st.sidebar.button("Apply Blur"):
            blurred_image = cv2.GaussianBlur(img_cv2, (blur_radius, blur_radius), 0)
            st.image(blurred_image, caption='Gambar setelah blur', channels='BGR', use_column_width=True)

    elif option == 'Pseudocolor':
        colormap = st.sidebar.selectbox("Pilih colormap", ['Jet', 'Hot', 'Cool', 'Spring'])
        if st.sidebar.button("Apply Pseudocolor"):
            gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            if colormap == 'Jet':
                pseudo_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
            elif colormap == 'Hot':
                pseudo_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_HOT)
            elif colormap == 'Cool':
                pseudo_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_COOL)
            elif colormap == 'Spring':
                pseudo_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_SPRING)
            st.image(pseudo_image, caption='Gambar setelah pseudocolor', channels='BGR', use_column_width=True)

    elif option == 'True Color Filter':
        color_filter = st.sidebar.selectbox("Filter warna", ['Red', 'Green', 'Blue'])
        if st.sidebar.button("Apply True Color Filter"):
            color_img = img_cv2.copy()
            if color_filter == 'Red':
                color_img[:, :, 1] = 0  # Hapus kanal hijau
                color_img[:, :, 2] = 0  # Hapus kanal biru
            elif color_filter == 'Green':
                color_img[:, :, 0] = 0  # Hapus kanal merah
                color_img[:, :, 2] = 0  # Hapus kanal biru
            elif color_filter == 'Blue':
                color_img[:, :, 0] = 0  # Hapus kanal merah
                color_img[:, :, 1] = 0  # Hapus kanal hijau
            st.image(color_img, caption=f'Gambar setelah filter {color_filter}', channels='BGR', use_column_width=True)

    elif option == 'Edge Detection (Prewitt, Sobel, Canny)':
        method = st.sidebar.selectbox("Metode deteksi tepi", ['Prewitt', 'Sobel', 'Canny'])
        if method == 'Prewitt':
            if st.sidebar.button("Apply Prewitt"):
                gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                prewitt_x = cv2.filter2D(gray_image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
                prewitt_y = cv2.filter2D(gray_image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
                prewitt_image = cv2.add(prewitt_x, prewitt_y)
                st.image(prewitt_image, caption='Gambar setelah Prewitt', channels='GRAY', use_column_width=True)

        elif method == 'Sobel':
            if st.sidebar.button("Apply Sobel"):
                gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
                sobel_image = cv2.magnitude(sobel_x, sobel_y)
                st.image(sobel_image, caption='Gambar setelah Sobel', channels='GRAY', use_column_width=True)

        elif method == 'Canny':
            threshold1 = st.sidebar.slider("Threshold 1 untuk Canny", 0, 255, 100)
            threshold2 = st.sidebar.slider("Threshold 2 untuk Canny", 0, 255, 200)
            if st.sidebar.button("Apply Canny"):
                canny_image = cv2.Canny(img_cv2, threshold1, threshold2)
                st.image(canny_image, caption='Gambar setelah Canny', channels='GRAY', use_column_width=True)

    # Simpan hasil
    st.write("Simpan hasil:")
    save_format = st.selectbox("Pilih format penyimpanan", ['JPEG', 'PNG'])
    save_image(image, save_format)