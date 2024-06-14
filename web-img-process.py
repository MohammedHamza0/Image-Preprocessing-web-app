import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import speech_recognition as sr
import pyttsx3



# # CSS to set background image
# page_bg_img = '''
# <style> 
# .stApp {
#   background-image: url("https://www.peak24.com.au/wp-content/uploads/2015/05/dark-backgrounds-15-cool-hd-background-and-wallpaper.jpg");
#   background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)




def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Image processing functions
def plot_histogram(image, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    if len(image.shape) == 3:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
    else:
        # If the image is grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='k')
        plt.xlim([0, 256])
    
    plt.grid(True)
    return plt

def display_with_histogram_original_image(original_image, title):
    original_array = np.array(original_image)
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    
    axes.imshow(original_array)
    axes.set_title("Original Image")
    axes.axis('off')
    
    hist_plt = plot_histogram(original_array, f"{title} Histogram")
    plt.show()
    st.pyplot(hist_plt)

def display_with_histogram_processed_image(processed_image, title):
    processed_array = np.array(processed_image)
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.imshow(processed_array)
    axes.set_title("Processed Image")
    axes.axis('off')
    
    hist_plt = plot_histogram(processed_image, f"{title} Histogram")
    plt.show()
    st.pyplot(hist_plt)

def complement_image(img):
    return 255 - img

def stretch_histogram(img_gray): 
    min_val = np.min(img_gray)
    max_val = np.max(img_gray)
    stretched = (img_gray - min_val) * (255.0 / (max_val - min_val))
    return stretched.astype(np.uint8)  

def histogram_equalization(img_gray): 
    return cv2.equalizeHist(img_gray)

def split_color_channels(img):
    b, g, r = cv2.split(img)
    zero_channel = np.zeros_like(b)
    return {
        "red": cv2.merge([b, g, zero_channel]),
        "green": cv2.merge([b, zero_channel, r]),
        "blue": cv2.merge([zero_channel, g, r]),
    }

# averaging filter 
def avg_blur(img, kernel_size=5):
    return cv2.blur(img, (kernel_size, kernel_size))

# Weighted Average Filter 
def g_blur(img, kernel_size=5):                           
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def adaptive_thresholding(gray_image, kernel_size=3):
    old_threshold = 128
    new_threshold = 0
    while True:
        blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        ret, binary1 = cv2.threshold(blurred_image, old_threshold, 255, cv2.THRESH_BINARY)
        ret, binary2 = cv2.threshold(blurred_image, old_threshold, 255, cv2.THRESH_BINARY_INV)
        m1 = np.mean(gray_image[binary1 > 0])
        m2 = np.mean(gray_image[binary2 > 0])
        new_threshold = 0.5 * (m1 + m2)  
        if abs(new_threshold - old_threshold) > 1:
            break
        old_threshold = new_threshold
    ret, binary = cv2.threshold(gray_image, new_threshold, 255, cv2.THRESH_BINARY)
    return binary

def adaptive_thresholding_rgb(rgb_image, kernel_size=3):
    channels = cv2.split(rgb_image)
    thresholded_channels = [adaptive_thresholding(channel, kernel_size) for channel in channels]
    thresholded_rgb = cv2.merge(thresholded_channels)
    return thresholded_rgb

def laplacian_filter(img, kernel_size=3):
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
    laplacian_8u = cv2.convertScaleAbs(laplacian)
    return laplacian_8u 

def solarize(img_gray, threshold=128):
    solarized_image = img_gray.copy()    
    solarized_image[solarized_image < threshold] = 255 - solarized_image[solarized_image < threshold]
    return solarized_image

def min_filter(img_gray, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    return cv2.erode(img_gray, kernel, iterations=1)

def max_filter(img_gray, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img_gray, kernel, iterations=1)

def swap_color_channels(img):
    b, g, r = cv2.split(img)
    return {                
        "Swap Red to Green": cv2.merge([b, r, g]),  
        "Swap Red to Blue": cv2.merge([r, g, b]),  
        "Swap Green to Blue": cv2.merge([g, b, r]),  
    }

def median_blur(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

def roberts_operator(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    img_x = cv2.filter2D(img_gray,-1 , kernel_x)
    img_y = cv2.filter2D(img_gray, -1, kernel_y)
    img_roberts = np.sqrt(np.square(img_x) + np.square(img_y))
    img_roberts = np.clip(img_roberts, 0, 255).astype(np.uint8)
    return cv2.convertScaleAbs(img_roberts)

def prewitt_operator(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    img_x = cv2.filter2D(img_gray, -1, kernel_x)
    img_y = cv2.filter2D(img_gray, -1, kernel_y)
    img_prewitt = np.sqrt(np.square(img_x) + np.square(img_y))
    img_prewitt[np.isnan(img_prewitt)] = 0
    img_prewitt[np.isinf(img_prewitt)] = 0
    img_prewitt_8bit = np.clip(img_prewitt, 0, 255).astype(np.uint8)
    return cv2.convertScaleAbs(img_prewitt_8bit)

def sobel_operator(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    sobel_magnitude[np.isnan(sobel_magnitude)] = 0
    sobel_magnitude[np.isinf(sobel_magnitude)] = 0
    return cv2.convertScaleAbs(np.clip(sobel_magnitude, 0, 255))

def dilate_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def erode_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def opening_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def internal_boundary(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    return cv2.subtract(img, eroded)

def external_boundary(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    return cv2.subtract(dilated, img)

def morphological_gradient(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def ensure_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image



# Function to display photo and professional message
def display_photo_and_message():
    st.markdown("<h2 style='text-align: center; color: #FF5733; font-size: 24px;'>Hello! I'm Mohammed Hamza Moawad Khalifa</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ffffff; font-size: 20px;'>A dedicated student at Menofia University's Faculty of Artificial Intelligence, specializing in Machine Intelligence. My passion for machine learning and deep learning drives me to push the boundaries of AI. Proficient in Python, I am eager to contribute to innovative projects and advance the field of artificial intelligence.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ffffff; font-size: 20px;'>Beyond technical expertise, I bring strong communication, leadership, and teamwork skills, ensuring effective collaboration and project success. With a commitment to continuous improvement and a knack for time management, I'm excited to engage in transformative AI endeavors.</p>", unsafe_allow_html=True)

# Custom HTML and CSS for the button styling
button_style = """
    <style>
        .button {
            background-color: #1e6091;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #155d80;
        }
    </style>"""


def main():

    # Authentication
    def login():
        # CSS to set background image
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("https://www.app.com.pk/wp-content/uploads/2023/12/Al-Aqsa-Mosque.jpg");
        background-size: cover;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.subheader(" اللهم ارزق إخواننا في فلسطين الصمود والقوة في وجه الطغيان وانصرهم")
        st.title("Login Page")
        username = st.text_input("# Username", "Gaza")
        password = st.text_input("# Password", "123" ,type="password")

        if st.button("Login"):
            if username == "Gaza" and password == "123":
                st.session_state['logged_in'] = True
            else:
                st.error("Invalid username or password")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login()
    else:
        st.title("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ")
        st.title("Image Processing Web App")

        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "gif"])


        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            st.subheader("Original image")
            st.image(image, caption='Uploaded Image', use_column_width=True)
            display_with_histogram_original_image(img_cv, "original")
            
            st.sidebar.subheader("إن الله وملائكته يصلون على النبي يا أيها الذين آمنوا صلوا عليه وسلموا تسليمًا")
            st.sidebar.subheader("Select a filter to apply to the image:")

            filter_choice = st.sidebar.selectbox("Select a filter", ["None",
                "Laplacian",
                "Gaussian Blur",
                "Adaptive Threshold",
                "Adaptive_thresholding_rgb",
                "Min Filter",
                "Max Filter",
                "Stretch Histogram",
                "Histogram Equalization",
                "Eliminate Color Channels",
                "Swap Color Channels",
                "Median Blur",
                "Complement",
                "Solarization",
                "Average Filter",
                "Prewitt Operator",
                "Sobel Operator",
                "Roberts Operator",
                "Dilation",
                "Erosion",
                "Opening image",
                "Closing image",
                "Internal boundary",
                "External boundary",
                "Morphological gradient"
            ])
            
            if filter_choice == "None":
                pass
            
            elif filter_choice == "Roberts Operator":
                processed_image = roberts_operator(img_cv)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Laplacian":
                kernel_size = st.sidebar.slider("Kernel size for Laplacian", 3, 7, step=2)
                processed_image = laplacian_filter(img_cv, kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Sobel Operator":
                processed_image = sobel_operator(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Prewitt Operator":
                processed_image = prewitt_operator(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Gaussian Blur":
                kernel_size = st.sidebar.slider("Kernel size for Gaussian Blur", 3, 11, step=2)
                processed_image = g_blur(img_cv, kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Complement":
                processed_image = complement_image(img_cv)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Adaptive Threshold":
                kernel_size = st.sidebar.slider("Kernel size for Adaptive Threshold", 3, 11, step=2)
                processed_image = adaptive_thresholding(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Adaptive_thresholding_rgb":
                kernel_size = st.sidebar.slider("Kernel size for Adaptive Threshold", 3, 11, step=2)
                processed_image = adaptive_thresholding_rgb(img_cv, kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Min Filter":
                kernel_size = st.sidebar.slider("Kernel size for Min Filter", 3, 11, step=2)
                processed_image = min_filter(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Dilation":
                kernel_size = st.sidebar.slider("Kernel size for dilate image", 3, 11, step=2)
                processed_image = dilate_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Erosion":
                kernel_size = st.sidebar.slider("Kernel size for erode image", 3, 11, step=2)
                processed_image = erode_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Opening image":
                kernel_size = st.sidebar.slider("Kernel size for opening image", 3, 11, step=2)
                processed_image = opening_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Closing image":
                kernel_size = st.sidebar.slider("Kernel size for closing image", 3, 11, step=2)
                processed_image = closing_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Internal boundary":
                kernel_size = st.sidebar.slider("Kernel size for internal boundary", 3, 11, step=2)
                processed_image = internal_boundary(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "External boundary":
                kernel_size = st.sidebar.slider("Kernel size for external boundary", 3, 11, step=2)
                processed_image = external_boundary(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Morphological gradient":
                kernel_size = st.sidebar.slider("Kernel size for morphological gradient", 3, 11, step=2)
                processed_image = morphological_gradient(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Max Filter":
                kernel_size = st.sidebar.slider("Kernel size for Max Filter", 3, 11, step=2)
                processed_image = max_filter(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Stretch Histogram":
                processed_image = stretch_histogram(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Histogram Equalization":
                processed_image = histogram_equalization(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Solarization":
                threshold = st.sidebar.slider("Threshold for Solarization", 0, 255, 128)
                processed_image = solarize(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), threshold)
                st.sidebar.success(f"The {filter_choice} filter applied.")
                
            elif filter_choice == "Eliminate Color Channels":
                color_channel = st.sidebar.radio("Select color channel", ["red", "green", "blue"])
                split_channels = split_color_channels(img_cv)
                processed_image = split_channels[color_channel]
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Swap Color Channels":
                channel_swap = st.sidebar.radio("Select channel swap", ["Swap Red to Green", "Swap Red to Blue", "Swap Green to Blue"])
                channel_swaps = swap_color_channels(img_cv)
                processed_image = channel_swaps[channel_swap]
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Median Blur":
                kernel_size = st.sidebar.slider("Kernel size for Median Blur", 3, 11, step=2)
                processed_image = median_blur(img_cv, kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                
            elif filter_choice == "Average Filter":
                kernel_size = st.sidebar.slider("Kernel size for Average Filter", 3, 11, step=2)
                processed_image = avg_blur(img_cv, kernel_size)
                st.sidebar.success(f"The {filter_choice} choice applied.")
                

            st.sidebar.subheader("Choose the mode to apply on the processed image.")
            display_mode = st.sidebar.radio("Display mode", ["RGB", "Grayscale", "Binary"])

            if display_mode == "Grayscale":
                processed_image = ensure_grayscale(processed_image)
            if display_mode == "Binary":
                processed_image = cv2.threshold(ensure_grayscale(processed_image), 128, 255, cv2.THRESH_BINARY)[1]
            
            
            
            if filter_choice == "None":
                pass
            else:
                st.subheader(f'Image after {filter_choice}')
                st.image(processed_image, caption=f'Image after {filter_choice}', use_column_width=True)
                display_with_histogram_processed_image(processed_image, 'processed')

                buffered = BytesIO()
                Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)).save(buffered, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=buffered.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
            st.markdown("---")
            # st.markdown("<h3 style='color: #336699;'>Designed and developed by <span style='color: #FF5733; font-weight: bold;'>Mohammed Hamza</span></h3>", unsafe_allow_html=True)
            # Sidebar container with button to display photo and professional message
            if st.sidebar.button("About me", key="about_me"):
                display_photo_and_message()
                st.markdown("<h2 style='color: #1e6091; font-weight: bold; font-family: sans-serif;'>Contact Me</h2>", unsafe_allow_html=True)
                st.markdown("""
                    <style>
                        .social-icon {
                            display: inline-block;
                            width: 30px;
                            height: 30px;
                            margin-right: 10px;
                            cursor: pointer;
                        }
                        .linkedin-logo {
                            background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/600px-LinkedIn_logo_initials.png');
                            background-size: cover;
                        }
                        .whatsapp-logo {
                            background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/WhatsApp.svg/800px-WhatsApp.svg.png');
                            background-size: cover;
                        }
                        .kaggle-logo {
                            background-image: url('https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png');
                            background-size: 90%;
                            background-position: center;
                        }
                        .github-logo {
                            background-image: url('https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg');
                            background-size: cover;
                            filter: invert(100%);
                        }
                        .social-link {
                            font-size: 18px;
                            color: #666;
                            text-decoration: none;
                        }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <a href='https://www.linkedin.com/in/mohammed-hamza-4184b2251/' target='_blank' class='social-link'>
                        <div class='social-icon linkedin-logo'></div>
                        LinkedIn
                    </a>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <a href='https://wa.me/201092317942' target='_blank' class='social-link'>
                        <div class='social-icon whatsapp-logo'></div>
                        WhatsApp
                    </a>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <a href='https://www.kaggle.com/mohammedhamzamoawad' target='_blank' class='social-link'>
                        <div class='social-icon kaggle-logo'></div>
                        Kaggle
                    </a>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <a href='https://github.com/MohammedHamza0' target='_blank' class='social-link'>
                        <div class='social-icon github-logo'></div>
                        GitHub
                    </a>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
