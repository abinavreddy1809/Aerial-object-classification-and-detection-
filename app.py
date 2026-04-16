# ============================================================
#  Streamlit App — Bird vs Drone Classifier 🦅 vs 🚁
#  Author : [Your Name] | Intern @ Labmentix
#  Run    : streamlit run app.py
#  Note   : This is my first Streamlit app ever. I followed
#            the docs and it actually worked!! 😄
# ============================================================

# Install required packages first:
# pip install streamlit tensorflow pillow numpy matplotlib ultralytics opencv-python

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import os
import time
import tempfile
import warnings
warnings.filterwarnings('ignore')

# ---- Tensorflow import (a little slow, that's normal) ----
import tensorflow as tf

# ---- Page config — must be first Streamlit command ----
st.set_page_config(
    page_title="Bird vs Drone Classifier | Labmentix",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================================
#  STYLING
#  I spent way too long on this CSS but it looks nice now 😊
# ==================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00d2ff, #a445b2, #ff4757);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
        letter-spacing: 1px;
    }

    .sub-title {
        text-align: center;
        color: #a8a8b3;
        font-size: 1rem;
        margin-bottom: 30px;
    }

    /* Prediction card */
    .pred-card {
        background: rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        text-align: center;
        margin: 10px 0;
    }

    /* Result text */
    .result-bird {
        font-size: 2.5rem;
        font-weight: 900;
        color: #2ed573;
    }

    .result-drone {
        font-size: 2.5rem;
        font-weight: 900;
        color: #ff4757;
    }

    /* Confidence bar */
    .conf-label {
        color: #a8a8b3;
        font-size: 0.9rem;
        margin-top: 8px;
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(0,0,0,0.3) !important;
    }

    /* Info box */
    .info-box {
        background: rgba(0,210,255,0.1);
        border-left: 4px solid #00d2ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.9rem;
    }

    /* Warning box */
    .warn-box {
        background: rgba(255,71,87,0.1);
        border-left: 4px solid #ff4757;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.9rem;
    }

    /* Upload section */
    .upload-section {
        border: 2px dashed rgba(0, 210, 255, 0.4);
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d2ff;
    }

    .metric-label {
        color: #a8a8b3;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ==================================================================
#  HELPER FUNCTIONS
# ==================================================================

@st.cache_resource   # cache so we don't reload the model on every interaction
def load_classification_model(model_path):
    """
    Loads the trained Keras model.
    @st.cache_resource means it loads once and stays in memory.
    Took me a while to figure this out — without it the page reloads forever!
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Resizes the image, converts to RGB (in case it's RGBA/grayscale),
    normalises pixel values, and adds a batch dimension.
    """
    img = image.convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0    # normalise to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)           # add batch dim: (1, 224, 224, 3)
    return img_array


def predict_image(model, image: Image.Image):
    """
    Runs inference and returns:
    - label    : 'Bird' or 'Drone'
    - confidence: float (0 to 1)
    - raw_prob  : the raw sigmoid output
    """
    img_array = preprocess_image(image)
    raw_prob = float(model.predict(img_array, verbose=0)[0][0])

    # In Keras binary classification:
    # output close to 0 = class 0 (bird), close to 1 = class 1 (drone)
    # (depends on how ImageDataGenerator sorted the folders alphabetically)
    if raw_prob >= 0.5:
        label = "Drone 🚁"
        confidence = raw_prob
    else:
        label = "Bird 🦅"
        confidence = 1 - raw_prob

    return label, confidence, raw_prob


def plot_confidence_bar(bird_conf, drone_conf):
    """
    A simple horizontal bar chart showing probabilities for both classes.
    """
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    classes = ['Bird 🦅', 'Drone 🚁']
    probs   = [bird_conf, drone_conf]
    colors  = ['#2ed573' if p == max(probs) else '#555577' for p in probs]

    bars = ax.barh(classes, probs, color=colors, height=0.5, edgecolor='none')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability', color='#a8a8b3', fontsize=10)
    ax.tick_params(colors='#a8a8b3')

    for bar, val in zip(bars, probs):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', color='white', fontweight='bold', fontsize=11)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


# ==================================================================
#  YOLO DETECTION (Optional — only loads if model file exists)
# ==================================================================

@st.cache_resource
def load_yolo_model(yolo_path):
    try:
        from ultralytics import YOLO
        model = YOLO(yolo_path)
        return model, None
    except ImportError:
        return None, "ultralytics not installed. Run: pip install ultralytics"
    except Exception as e:
        return None, str(e)


def run_yolo_detection(yolo_model, image: Image.Image, conf_threshold=0.5):
    """
    Runs YOLOv8 inference and returns the image with bounding boxes drawn.
    """
    # Save PIL image to a temp file (YOLO needs a file path)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    results = yolo_model.predict(
        source=tmp_path,
        conf=conf_threshold,
        verbose=False
    )
    os.unlink(tmp_path)  # clean up temp file

    # Draw boxes using matplotlib
    result = results[0]
    img_array = np.array(image.convert('RGB'))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f0c29')
    ax.set_facecolor('#0f0c29')
    ax.imshow(img_array)
    ax.axis('off')

    class_names = ['Bird', 'Drone']
    colors_map  = {'Bird': '#2ed573', 'Drone': '#ff4757'}

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf   = float(box.conf[0].cpu().numpy())
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f'Class {cls_id}'
            color = colors_map.get(cls_name, '#00d2ff')

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'{cls_name} {conf*100:.0f}%',
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    else:
        ax.text(img_array.shape[1]//2, img_array.shape[0]//2,
                'No objects detected', color='white', fontsize=14,
                ha='center', va='center',
                bbox=dict(facecolor='#333', alpha=0.7, boxstyle='round'))

    plt.tight_layout(pad=0)
    return fig, len(result.boxes) if result.boxes is not None else 0


# ==================================================================
#  SIDEBAR
# ==================================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    # Model paths
    st.markdown("### 📁 Model Files")
    clf_model_path = st.text_input(
        "Classification Model Path",
        value="best_aerial_model.keras",
        help="Path to your trained .keras model file"
    )

    yolo_model_path = st.text_input(
        "YOLOv8 Model Path (optional)",
        value="runs/detect/bird_drone_yolo/weights/best.pt",
        help="Leave this as-is if you haven't trained YOLO yet"
    )

    st.markdown("---")
    st.markdown("### 🎛️ Detection Settings")
    conf_threshold = st.slider(
        "YOLO Confidence Threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Lower = more detections, Higher = more precise"
    )

    enable_yolo = st.checkbox(
        "Enable YOLOv8 Detection",
        value=False,
        help="Enable only if you have a trained YOLO model"
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style='color: #a8a8b3; font-size: 0.82rem; line-height: 1.6;'>
    <b>Project:</b> Aerial Object Classification<br>
    <b>Intern:</b> [Your Name]<br>
    <b>Company:</b> Labmentix<br>
    <b>Models:</b> Custom CNN + ResNet50<br>
    <b>Classes:</b> Bird 🦅 / Drone 🚁
    </div>
    """, unsafe_allow_html=True)


# ==================================================================
#  MAIN PAGE
# ==================================================================

# Header
st.markdown('<div class="main-title">🛸 Bird vs Drone Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Aerial Object Classification | Labmentix Data Science Internship Project</div>',
            unsafe_allow_html=True)

st.markdown("---")

# Load models
clf_model, clf_error = load_classification_model(clf_model_path)

if clf_error:
    st.markdown(f"""
    <div class="warn-box">
    ⚠️ <b>Model not found:</b> {clf_error}<br><br>
    Make sure <code>best_aerial_model.keras</code> is in the same folder as <code>app.py</code>.
    Download it from your Google Colab output files panel.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box">
    ✅ <b>Classification model loaded successfully!</b>
    Upload an aerial image below to classify it as Bird or Drone.
    </div>
    """, unsafe_allow_html=True)

# YOLO model
yolo_model = None
if enable_yolo:
    yolo_model, yolo_error = load_yolo_model(yolo_model_path)
    if yolo_error:
        st.warning(f"⚠️ YOLO model issue: {yolo_error}")
    else:
        st.success("✅ YOLOv8 detection model loaded!")

st.markdown("---")

# ==================================================================
#  TABS : Single Image | Batch | About
# ==================================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Image Classification",
    "📂 Batch Classification",
    "📊 About the Project"
])


# ------------------------------------------------------------------
# TAB 1 : SINGLE IMAGE CLASSIFICATION
# ------------------------------------------------------------------
with tab1:
    st.markdown("### Upload an Aerial Image")
    st.markdown("Supported formats: **JPG, JPEG, PNG**")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        key='single_upload'
    )

    if uploaded_file is not None and clf_model is not None:
        image = Image.open(uploaded_file)

        # Layout: image on left, results on right
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("**Uploaded Image:**")
            st.image(image, use_column_width=True, caption=uploaded_file.name)
            st.caption(f"Size: {image.size[0]} × {image.size[1]} px | Mode: {image.mode}")

        with col_result:
            st.markdown("**Classification Result:**")

            with st.spinner("🤔 Thinking..."):
                time.sleep(0.3)  # tiny delay so the spinner shows
                label, confidence, raw_prob = predict_image(clf_model, image)

            # Bird or Drone confidence
            drone_conf = raw_prob
            bird_conf  = 1 - raw_prob

            # Result card
            result_class = "result-bird" if "Bird" in label else "result-drone"
            st.markdown(f"""
            <div class="pred-card">
                <div class="{result_class}">{label}</div>
                <div style="font-size: 1.1rem; color: #f0f0f0; margin-top: 8px;">
                    Confidence: <b>{confidence*100:.1f}%</b>
                </div>
                <div class="conf-label">
                    {"High confidence ✅" if confidence > 0.85 else
                     "Moderate confidence ⚠️" if confidence > 0.65 else
                     "Low confidence ❌ — try a clearer image"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar chart
            st.markdown("**Probability Breakdown:**")
            fig_bar = plot_confidence_bar(bird_conf, drone_conf)
            st.pyplot(fig_bar, use_container_width=True)
            plt.close()

        # Optional YOLO detection
        if enable_yolo and yolo_model is not None:
            st.markdown("---")
            st.markdown("### 🎯 YOLOv8 Object Detection")

            with st.spinner("Running YOLOv8 detection..."):
                det_fig, n_detections = run_yolo_detection(
                    yolo_model, image, conf_threshold
                )

            st.markdown(f"**Detections found: {n_detections}**")
            st.pyplot(det_fig, use_container_width=True)
            plt.close()

    elif uploaded_file is None:
        st.markdown("""
        <div class="upload-section">
            <div style="font-size: 3rem;">📸</div>
            <div style="color: #a8a8b3; margin-top: 10px;">
                Drag and drop an image here, or click <b>Browse files</b>
            </div>
            <div style="color: #555577; font-size: 0.85rem; margin-top: 5px;">
                Works best with clear aerial photos of birds or drones
            </div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# TAB 2 : BATCH CLASSIFICATION
# ------------------------------------------------------------------
with tab2:
    st.markdown("### Upload Multiple Images")
    st.markdown("Classify several images at once and see results in a table.")

    batch_files = st.file_uploader(
        "Choose multiple images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key='batch_upload'
    )

    if batch_files and clf_model is not None:
        st.markdown(f"**{len(batch_files)} image(s) uploaded. Running classification...**")

        # Progress bar (this took me a while to figure out)
        progress = st.progress(0)
        results_data = []

        for i, file in enumerate(batch_files):
            img = Image.open(file)
            label, confidence, raw_prob = predict_image(clf_model, img)
            results_data.append({
                'File Name':   file.name,
                'Prediction':  label,
                'Confidence':  f"{confidence*100:.1f}%",
                'Bird Prob':   f"{(1-raw_prob)*100:.1f}%",
                'Drone Prob':  f"{raw_prob*100:.1f}%",
            })
            progress.progress((i + 1) / len(batch_files))

        progress.empty()  # remove progress bar when done

        # Show results table
        import pandas as pd
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)

        # Summary stats
        bird_count  = sum(1 for r in results_data if 'Bird'  in r['Prediction'])
        drone_count = sum(1 for r in results_data if 'Drone' in r['Prediction'])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(batch_files)}</div>
                <div class="metric-label">Total Images</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#2ed573">{bird_count} 🦅</div>
                <div class="metric-label">Birds Detected</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#ff4757">{drone_count} 🚁</div>
                <div class="metric-label">Drones Detected</div>
            </div>""", unsafe_allow_html=True)

        # Pie chart
        if bird_count + drone_count > 0:
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            fig_pie.patch.set_facecolor('#1a1a2e')
            ax_pie.set_facecolor('#1a1a2e')
            ax_pie.pie(
                [bird_count, drone_count],
                labels=['Bird', 'Drone'],
                colors=['#2ed573', '#ff4757'],
                autopct='%1.0f%%',
                textprops={'color': 'white', 'fontsize': 12},
                startangle=90
            )
            ax_pie.set_title('Batch Results', color='white', fontsize=13)
            st.pyplot(fig_pie, use_container_width=False)
            plt.close()

        # Download results as CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_data,
            file_name="classification_results.csv",
            mime="text/csv"
        )

    elif not batch_files:
        st.info("Upload multiple images using the uploader above to classify them in bulk.")


# ------------------------------------------------------------------
# TAB 3 : ABOUT THE PROJECT
# ------------------------------------------------------------------
with tab3:
    st.markdown("### 📋 About This Project")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        **Project Title:** Aerial Object Classification & Detection

        **Problem Statement:**
        This app classifies aerial images into two categories — **Bird** or **Drone** —
        to support security surveillance, wildlife protection, and airspace safety.

        **Why is this important?**
        - 🛡️ Security: Identify unauthorized drones in restricted zones
        - 🐦 Wildlife: Monitor bird populations near wind farms
        - ✈️ Airports: Detect birds near runways to prevent strikes
        - 🌿 Research: Track animals using aerial footage accurately
        """)

    with col_right:
        st.markdown("""
        **Dataset:**
        - Train: 1,414 birds + 1,248 drones
        - Validation: 217 birds + 225 drones
        - Test: 121 birds + 94 drones

        **Models Trained:**
        - Custom CNN (built from scratch)
        - ResNet50 Transfer Learning (fine-tuned)
        - YOLOv8 (optional, for object detection)

        **Tech Stack:**
        TensorFlow/Keras, YOLOv8, Streamlit, Python
        """)

    st.markdown("---")
    st.markdown("""
    **Project Workflow:**

    1. 🔍 Dataset exploration & visualisation
    2. 🔧 Preprocessing (resize to 224×224, normalise to [0,1])
    3. 🔄 Data Augmentation (rotation, flip, zoom, brightness)
    4. 🧠 Model Building (Custom CNN + Transfer Learning)
    5. 📈 Training with EarlyStopping & ModelCheckpoint
    6. 📊 Evaluation (Confusion Matrix, F1-Score, Accuracy curves)
    7. 🏆 Best model selected and deployed here!
    8. (Optional) 🎯 YOLOv8 for real-time bounding box detection
    """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#a8a8b3; font-size:0.85rem; padding:10px 0;'>
    Built with ❤️ during my internship at <b>Labmentix</b><br>
    Technologies: Python · TensorFlow · YOLOv8 · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ==================================================================
#  FOOTER
# ==================================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555577; font-size:0.8rem; padding:8px 0;'>
    Labmentix Data Science Internship Project — Aerial Object Classification
</div>
""", unsafe_allow_html=True)
